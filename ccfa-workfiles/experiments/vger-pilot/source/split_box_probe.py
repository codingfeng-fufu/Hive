"""Measure the split decision with annotated crops and audit retrieved image support."""

import argparse
import json
from collections import Counter
from pathlib import Path

import fitz
import yaml
from PIL import Image

from eval.vger_pilot import _crop, _label
from src.config import load_config
from src.embeddings import SiglipEncoder
from src.layout import native_region


def score_boxes(annotations: dict, pdfs: Path, cfg: Path) -> dict:
    settings = load_config(cfg)
    encoder = SiglipEncoder(settings.encoder)
    rows = []
    for name, raw in annotations.items():
        occurrences = raw["occurrences"]
        with fitz.open(pdfs / f"{name}.pdf") as source:
            images = [_crop(source, page, box) for page, box, _, _ in occurrences.values()]
            vectors = dict(zip(occurrences, encoder.image(images), strict=True))
        for left, right in raw["pairs"]:
            same = occurrences[left][3] == occurrences[right][3]
            similarity = float(vectors[left] @ vectors[right])
            rows.append(
                {
                    "document": name,
                    "left": left,
                    "right": right,
                    "same_identity": same,
                    "visual_cosine": similarity,
                    "would_merge": similarity >= settings.graph.tau_split,
                }
            )
    summary = {
        "tau_split": settings.graph.tau_split,
        "pairs": len(rows),
        "negative_total": sum(not row["same_identity"] for row in rows),
        "false_merges": sum(not row["same_identity"] and row["would_merge"] for row in rows),
        "positive_total": sum(row["same_identity"] for row in rows),
        "same_identity_merges": sum(row["same_identity"] and row["would_merge"] for row in rows),
    }
    return {"summary": summary, "pairs": rows}


def _fraction_covered(box: tuple, evidence: tuple) -> float:
    width = max(0, min(box[2], evidence[2]) - max(box[0], evidence[0]))
    height = max(0, min(box[3], evidence[3]) - max(box[1], evidence[1]))
    return width * height / ((box[2] - box[0]) * (box[3] - box[1]))


def score_retrieval(annotations: dict, run: Path) -> dict:
    payload = json.loads((run / "results.json").read_text())
    gold = {}
    for name, raw in annotations.items():
        sizes = {}
        for page, _, _, _ in raw["occurrences"].values():
            if page not in sizes:
                with Image.open(run / "pages" / name / f"{page}.png") as raster:
                    sizes[page] = raster.size
        gold[name] = {
            key: (label, identity, native_region(page - 1, bbox, sizes[page], "normalized_1000"))
            for key, (page, bbox, label, identity) in raw["occurrences"].items()
        }
    rows, counts = [], {}
    for record in payload["records"]:
        name = record["document"]
        group = (record["family"], record["mode"])
        counter = counts.setdefault("/".join(group), Counter())
        for query in record["queries"]:
            label, identity, target = gold[name][query["anchor"]]
            images = [item for item in query["selected"] if item["include_image"] and item["bbox"]]
            target_visible = False
            wrong_visible = False
            for item in images:
                if (
                    item["page"] == target.page
                    and _fraction_covered(target.bbox, item["bbox"]) >= 0.8
                ):
                    target_visible = True
                for other_label, other_id, region in gold[name].values():
                    if (
                        _label(other_label) == _label(label)
                        and other_id != identity
                        and item["page"] == region.page
                        and _fraction_covered(region.bbox, item["bbox"]) >= 0.8
                    ):
                        wrong_visible = True
            counter.update(
                {
                    "queries": 1,
                    "queries_with_distractor": query["has_same_name_distractor"],
                    "target_visible": target_visible,
                    "wrong_visible": wrong_visible,
                    "wrong_only": wrong_visible and not target_visible,
                    "both_visible": wrong_visible and target_visible,
                }
            )
            rows.append(
                {
                    "document": name,
                    "family": group[0],
                    "mode": group[1],
                    "anchor": query["anchor"],
                    "target_visible": target_visible,
                    "wrong_visible": wrong_visible,
                }
            )
    return {"minimum_gold_area_covered": 0.8, "summary": counts, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--pdfs", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--retrieval-run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    annotations = yaml.safe_load(args.annotations.read_text())["documents"]
    if args.retrieval_run:
        result = score_retrieval(annotations, args.retrieval_run)
    else:
        if args.pdfs is None or args.config is None:
            parser.error("Pass --pdfs and --config for the supplied-box diagnostic")
        result = score_boxes(annotations, args.pdfs, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
