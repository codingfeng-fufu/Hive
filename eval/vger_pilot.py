"""Evaluate annotated entity pairs on real PDF excerpts."""

import argparse
import csv
import hashlib
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import fitz
import numpy as np
import yaml
from PIL import Image

from src.config import EncoderConfig, GraphConfig
from src.embeddings import SiglipEncoder
from src.layout import native_region, read_mineru

METHODS = ("text", "figure", "crop")
STRATA = (
    "same_name_different",
    "same_name_same",
    "different_name_same",
    "different_name_different",
)


@dataclass(frozen=True)
class Occurrence:
    entity: str
    label: str
    figure: str
    page: int
    bbox: tuple[int, int, int, int]
    figure_bbox: tuple[int, int, int, int]


def _label(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def load_pairs(annotations: dict, *, seed: int) -> dict[str, tuple[str, list]]:
    rng = random.Random(seed)
    output = {}
    for filename, raw in annotations["documents"].items():
        occurrences = []
        for figure, box, entity, labels in raw["regions"]:
            page, full_bbox, _ = raw["figures"][figure]
            if not (
                full_bbox[0] <= box[0] < box[2] <= full_bbox[2]
                and full_bbox[1] <= box[1] < box[3] <= full_bbox[3]
            ):
                raise ValueError(f"{filename} page {page}: {box} outside figure {full_bbox}")
            occurrences.extend(
                Occurrence(entity, label, figure, page, tuple(box), tuple(full_bbox))
                for label in labels
            )
        strata = defaultdict(list)
        for left, right in combinations(occurrences, 2):
            same_name = _label(left.label) == _label(right.label)
            same_entity = left.entity == right.entity
            name = (
                ("same_name_same" if same_entity else "same_name_different")
                if same_name
                else ("different_name_same" if same_entity else "different_name_different")
            )
            strata[name].append((left, right))
        selected = []
        quotas = dict(zip(STRATA, (3, 2, 3, 3)))
        for name in STRATA:
            pool = strata[name]
            rng.shuffle(pool)
            # Prefer distinct visual instances for coreference positives.
            if name in {"same_name_same", "different_name_same"}:
                pool.sort(
                    key=lambda pair: (
                        (pair[0].page, pair[0].bbox)
                        == (
                            pair[1].page,
                            pair[1].bbox,
                        )
                    )
                )
            selected.extend((name, *pair) for pair in pool[: quotas[name]])
        while len(selected) < 8:
            taken = len([pair for pair in selected if pair[0] == "different_name_different"])
            pool = strata["different_name_different"]
            if taken == len(pool):
                break
            selected.append(("different_name_different", *pool[taken]))
        output[filename] = (raw["split"], selected)
    return output


def _crop(pdf: fitz.Document, page: int, bbox: tuple, dpi: int = 250) -> Image.Image:
    source = pdf[page - 1]
    box = fitz.Rect(
        *[
            v * (source.rect.width if i % 2 == 0 else source.rect.height) / 1000
            for i, v in enumerate(bbox)
        ]
    )
    pixels = source.get_pixmap(dpi=dpi, clip=box, alpha=False)
    return Image.frombytes("RGB", (pixels.width, pixels.height), pixels.samples)


def _decision(row: dict, method: str, split: float, merge: float, tau_text: float) -> str:
    if row["text"] >= tau_text:
        return "merge" if method == "text" or row[method] >= split else "none"
    if method != "text" and row[method] >= merge:
        return "alias"
    return "none"


def _metrics(rows: list[dict], method: str) -> dict:
    negative = [row for row in rows if row["stratum"] == STRATA[0]]
    positive = [row for row in rows if row["stratum"] == STRATA[1]]
    aliases = [row for row in rows if row["stratum"] == STRATA[2]]
    predicted = [row for row in rows if row[method] == "alias"]
    correct = sum(row["stratum"] == STRATA[2] for row in predicted)
    return {
        "same_name_different_n": len(negative),
        "false_merge_n": sum(row[method] == "merge" for row in negative),
        "same_name_same_n": len(positive),
        "retained_n": sum(row[method] == "merge" for row in positive),
        "alias_true_n": len(aliases),
        "alias_predicted_n": len(predicted),
        "alias_correct_n": correct,
        "alias_precision": correct / len(predicted) if predicted else None,
        "alias_recall": correct / len(aliases) if aliases else None,
    }


def _threshold(rows: list[dict], method: str, graph: GraphConfig) -> tuple[float, float]:
    def quality(split, merge):
        decisions = [
            dict(row, prediction=_decision(row, method, split, merge, graph.tau_text))
            for row in rows
        ]
        metrics = _metrics(decisions, "prediction")
        balance = (
            1
            - metrics["false_merge_n"] / max(1, metrics["same_name_different_n"])
            + metrics["retained_n"] / max(1, metrics["same_name_same_n"])
        )
        precision = metrics["alias_precision"] or 0
        recall = metrics["alias_recall"] or 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        return (
            balance + f1,
            -round(abs(split - graph.tau_split) + abs(merge - graph.tau_merge), 5),
        )

    choices = [
        (split, merge)
        for split in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
        for merge in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)
    ]
    return max(choices, key=lambda choice: quality(*choice))


def _interval(rows: list[dict], field: str, stratum: str, seed: int) -> list[float] | None:
    clusters = defaultdict(list)
    for row in rows:
        if row["stratum"] == stratum:
            clusters[row["document"]].append(
                int(row["decision_crop"] == field) - int(row["decision_figure"] == field)
            )
    if len(clusters) < 2:
        return None
    rng = random.Random(seed)
    values = list(clusters.values())
    samples = []
    for _ in range(3000):
        batch = rng.choices(values, k=len(values))
        samples.append(sum(map(sum, batch)) / sum(map(len, batch)))
    return np.quantile(samples, [0.025, 0.975]).tolist()


def _overlap(left: tuple, right: tuple) -> float:
    width = max(0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = width * height
    return intersection / (
        (left[2] - left[0]) * (left[3] - left[1])
        + (right[2] - right[0]) * (right[3] - right[1])
        - intersection
    )


def coverage(
    pairs: dict,
    scores: list[dict],
    assets: Path,
    layout: Path,
    page_map: Path,
    cache: Path,
    tau_text: float,
) -> dict:
    mapping = json.loads(page_map.read_text())
    blocks_by_doc = defaultdict(list)
    pages_by_doc = defaultdict(set)
    for doc_name, page in mapping:
        pages_by_doc[doc_name].add(page)
    for block in json.loads(layout.read_text()):
        doc_name, page = mapping[block["page_idx"]]
        blocks_by_doc[doc_name].append({**block, "page_idx": page - 1})
    hit = {}
    cache.mkdir(parents=True, exist_ok=True)
    for name, (_, checked) in pairs.items():
        pages = {}
        with fitz.open(assets / f"{name}.pdf") as pdf:
            for page in sorted(pages_by_doc[name]):
                path = cache / f"{name}-{page}.png"
                pdf[page - 1].get_pixmap(dpi=160, alpha=False).save(path)
                pages[page - 1] = path
        if not blocks_by_doc[name]:
            raise ValueError(f"No MinerU blocks available for {name}")
        path = cache / f"{name}.json"
        path.write_text(json.dumps(blocks_by_doc[name]))
        document = read_mineru(name, pages, path, assets / f"{name}.pdf")
        anchors = {m for _, a, b in checked for m in (a, b)}
        for anchor in anchors:
            normalized_name = _label(anchor.label)
            source_text = " ".join(
                str(block.get("text", ""))
                for block in blocks_by_doc[name]
                if block["page_idx"] == anchor.page - 1
            )
            with Image.open(pages[anchor.page - 1]) as raster:
                target = native_region(
                    anchor.page - 1, list(anchor.bbox), raster.size, "normalized_1000"
                )
            markers = [
                item.region
                for item in document.mentions
                if item.region and item.region.page == target.page
            ]
            grounded = [
                item.region
                for item in document.mentions
                if item.kind == "entity"
                and item.region
                and item.region.page == target.page
                and _label(item.text) == _label(anchor.label)
            ]
            hit[(name, anchor)] = (
                any(_overlap(target.bbox, box.bbox) >= 0.5 for box in markers),
                any(_overlap(target.bbox, box.bbox) >= 0.5 for box in grounded),
                f" {normalized_name} " in f" {_label(source_text)} ",
                any(
                    item.kind == "entity"
                    and item.page == target.page
                    and _label(item.text) == normalized_name
                    for item in document.mentions
                ),
            )
    counts = defaultdict(int)
    offset = 0
    for name, (_, checked) in pairs.items():
        for stratum, left, right in checked:
            row = scores[offset]
            offset += 1
            left_hit, right_hit = hit[(name, left)], hit[(name, right)]
            row["automatic_both_regions"] = left_hit[0] and right_hit[0]
            row["automatic_both_entities"] = left_hit[1] and right_hit[1]
            row["both_names_in_mineru_same_page_text"] = left_hit[2] and right_hit[2]
            row["automatic_both_names"] = left_hit[3] and right_hit[3]
            if stratum in STRATA[:3]:
                counts[f"{stratum}_n"] += 1
                counts[f"{stratum}_both_entities_n"] += row["automatic_both_entities"]
                counts[f"{stratum}_both_and_text_candidate_n"] += row[
                    "automatic_both_entities"
                ] and ((row["text"] < tau_text) == (stratum == STRATA[2]))
    return {
        "labeled_occurrences": len(hit),
        "automatic_region_hits_iou_0_5": sum(value[0] for value in hit.values()),
        "automatic_named_grounded_hits_iou_0_5": sum(value[1] for value in hit.values()),
        "names_in_mineru_same_page_text": sum(value[2] for value in hit.values()),
        "automatic_name_hits": sum(value[3] for value in hit.values()),
        **counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--pdfs", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layout", type=Path)
    parser.add_argument("--page-map", type=Path)
    parser.add_argument("--cached-scores", type=Path)
    options = parser.parse_args()
    if bool(options.layout) != bool(options.page_map):
        parser.error("--layout requires --page-map and vice versa")
    annotation_bytes = options.annotations.read_bytes()
    annotations = yaml.safe_load(annotation_bytes)
    pairs = load_pairs(annotations, seed=options.seed)
    manifest_hash = hashlib.sha256(annotation_bytes).hexdigest()
    options.output.mkdir(parents=True, exist_ok=True)
    if options.cached_scores:
        saved = json.loads(options.cached_scores.read_text())
        if saved["sha256"] != manifest_hash or saved["seed"] != options.seed:
            raise ValueError("Raw scores do not match the annotations and seed")
        scores = saved["rows"]
    else:
        encoder = SiglipEncoder(
            EncoderConfig(model=options.model, device=options.device, batch_size=2)
        )
        labels = sorted(
            {item.label for _, checked in pairs.values() for _, a, b in checked for item in (a, b)}
        )
        text = dict(zip(labels, encoder.text(labels), strict=True))
        scores = []
        for name, (split, checked) in pairs.items():
            regions = sorted(
                {
                    (item.page, box)
                    for _, a, b in checked
                    for item in (a, b)
                    for box in (item.bbox, item.figure_bbox)
                }
            )
            with fitz.open(options.pdfs / f"{name}.pdf") as pdf:
                vectors = encoder.image([_crop(pdf, pg, box) for pg, box in regions])
            visual = dict(zip(regions, vectors, strict=True))
            for stratum, left, right in checked:
                row = {
                    "document": name,
                    "split": split,
                    "stratum": stratum,
                    "same_region": (left.page, left.bbox) == (right.page, right.bbox),
                }
                for side, item in (("left", left), ("right", right)):
                    row.update({f"{side}_{key}": value for key, value in vars(item).items()})
                row["text"] = float(text[left.label] @ text[right.label])
                for method, key in (("figure", "figure_bbox"), ("crop", "bbox")):
                    row[method] = float(
                        visual[left.page, getattr(left, key)]
                        @ visual[right.page, getattr(right, key)]
                    )
                scores.append(row)
            print(f"{name}: {len(checked)} pairs", flush=True)
        (options.output / "raw_scores.json").write_text(
            json.dumps(
                {"sha256": manifest_hash, "seed": options.seed, "rows": scores},
                indent=2,
            )
        )
    defaults = GraphConfig()
    automatic = (
        coverage(
            pairs,
            scores,
            options.pdfs,
            options.layout,
            options.page_map,
            options.output / "layout-cache",
            defaults.tau_text,
        )
        if options.layout
        else None
    )
    development = [row for row in scores if row["split"] == "development"]
    heldout = [row for row in scores if row["split"] == "heldout"]
    thresholds = {method: _threshold(development, method, defaults) for method in METHODS[1:]}
    thresholds["text"] = (defaults.tau_split, defaults.tau_merge)
    results = {}
    for setting in ("defaults", "development_thresholds"):
        for row in scores:
            for method in METHODS:
                split, merge = (
                    (defaults.tau_split, defaults.tau_merge)
                    if setting == "defaults"
                    else thresholds[method]
                )
                row[f"{setting}_{method}"] = _decision(row, method, split, merge, defaults.tau_text)
                row[f"decision_{method}"] = row[f"{setting}_{method}"]
        subsets = {
            "all": heldout,
            "distinct_region": [row for row in heldout if not row["same_region"]],
        }
        results[setting] = {
            subset: {method: _metrics(rows, f"decision_{method}") for method in METHODS}
            for subset, rows in subsets.items()
        }
        results[setting]["crop_minus_figure_95ci"] = {
            "false_merge_rate": _interval(heldout, "merge", STRATA[0], options.seed),
            "alias_recall": _interval(subsets["distinct_region"], "alias", STRATA[2], options.seed),
        }
    with (options.output / "pairs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=scores[0].keys())
        writer.writeheader()
        writer.writerows(scores)
    summary = {
        "annotation_sha256": manifest_hash,
        "seed": options.seed,
        "pdfs": len(pairs),
        "pairs": len(scores),
        "development_pairs": len(development),
        "heldout_pairs": len(heldout),
        "tau_text": defaults.tau_text,
        "calibrated_thresholds": thresholds,
        "heldout": results,
        "mineru_coverage": automatic,
    }
    (options.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
