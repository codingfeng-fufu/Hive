"""Measure visual evidence gains over PDF text on fixed annotated pairs."""

import argparse
import hashlib
import json
from itertools import groupby
from pathlib import Path

import fitz
import numpy as np
import torch
import yaml
from scipy.stats import rankdata
from transformers import AutoModel, AutoTokenizer

from eval.vger_pilot import STRATA, _crop, _label, _metrics
from src.config import EncoderConfig
from src.embeddings import SiglipEncoder

METHODS = ("scoped_text", "scoped_text_figure", "scoped_text_crop")


def page_lines(page):
    words = page.get_text("words", sort=False)
    result = []
    for _, group in groupby(words, key=lambda word: tuple(word[5:7])):
        line = list(group)
        box = fitz.Rect(line[0][:4])
        for word in line[1:]:
            box.include_rect(fitz.Rect(word[:4]))
        result.append((box, " ".join(word[4] for word in line)))
    return result


def rect(page, bbox):
    width, height = page.rect.width, page.rect.height
    return fitz.Rect(
        *(
            value * scale / 1000
            for value, scale in zip(bbox, (width, height, width, height), strict=True)
        )
    )


def distance(box, target):
    dx = max(target.x0 - box.x1, box.x0 - target.x1, 0)
    dy = max(target.y0 - box.y1, box.y0 - target.y1, 0)
    return dx + dy, abs(box.y0 - target.y0), abs(box.x0 - target.x0)


def source_context(lines, region, figure, label):
    local = sorted(lines, key=lambda line: distance(line[0], region))[:5]
    outside = (
        line
        for line in lines
        if not line[0].intersects(figure) and distance(line[0], figure)[0] <= 36
    )
    captions = sorted(outside, key=lambda line: distance(line[0], figure))[:3]
    return "\n".join(dict.fromkeys([label, *(line for _, line in (*local, *captions))]))


def embedding_lookup(contexts, model_path):
    texts = sorted(set(contexts))
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).eval()
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(texts), 16):
            batch = tokenizer(
                texts[start : start + 16],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            )
            vectors.extend(torch.nn.functional.normalize(model(**batch).last_hidden_state[:, 0]))
    return dict(zip(texts, vectors, strict=True))


def scores_for(rows, method, name_type):
    strata = STRATA[:2] if name_type == "same" else STRATA[2:]
    subset = [row for row in rows if row["stratum"] in strata]
    positive = STRATA[1] if name_type == "same" else STRATA[2]
    return np.array([row[method] for row in subset]), np.array(
        [row["stratum"] == positive for row in subset]
    )


def threshold(scores, positive, name_type):
    def objective(cutoff):
        hit = scores >= cutoff
        true_positive = (positive & hit).sum()
        if name_type == "same":
            return true_positive / positive.sum() + (~positive & ~hit).sum() / (~positive).sum()
        return 2 * true_positive / (positive.sum() + hit.sum())

    return float(max([*sorted(set(scores)), np.nextafter(scores.max(), np.inf)], key=objective))


def auc(scores, positive):
    n = positive.sum()
    ranks = rankdata(scores)
    return float((ranks[positive].sum() - n * (n + 1) / 2) / (n * (len(scores) - n)))


def score_reference_pairs(manifest, pdfs, vision_model, device):
    encoder = SiglipEncoder(EncoderConfig(model=vision_model, device=device, batch_size=2))
    for name, entry in manifest["documents"].items():
        figures, mentions = entry["figures"], entry["occurrences"]
        figures = {
            key: (entry["page"], tuple(box)) if "page" in entry else (box[0], tuple(box[1]))
            for key, box in figures.items()
        }
        boxes = sorted(
            set(figures.values())
            | {(figures[parent][0], tuple(box)) for parent, box, _, _ in mentions.values()}
        )
        with fitz.open(pdfs / f"{name}.pdf") as pdf:
            vectors = dict(
                zip(
                    boxes,
                    encoder.image([_crop(pdf, page, box) for page, box in boxes]),
                    strict=True,
                )
            )
        for pair in entry["pairs"]:
            row = {"document": name, "split": "reference_seed", "evidence": pair["evidence"]}
            for side in ("left", "right"):
                parent, box, label, entity = mentions[pair[side]]
                page, parent_box = figures[parent]
                row.update(
                    {
                        f"{side}_page": page,
                        f"{side}_bbox": box,
                        f"{side}_label": label,
                        f"{side}_figure_bbox": parent_box,
                        f"{side}_entity": entity,
                    }
                )
            same_name = _label(row["left_label"]) == _label(row["right_label"])
            same_entity = row["left_entity"] == row["right_entity"]
            if same_name:
                row["stratum"] = "same_name_same" if same_entity else "same_name_different"
            else:
                row["stratum"] = (
                    "different_name_same" if same_entity else "different_name_different"
                )
            for view, key in (("figure", "figure_bbox"), ("crop", "bbox")):
                row[view] = float(
                    vectors[row["left_page"], tuple(row[f"left_{key}"])]
                    @ vectors[row["right_page"], tuple(row[f"right_{key}"])]
                )
            yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--pdfs", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-pairs", type=Path)
    parser.add_argument("--vision-model")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if bool(args.reference_pairs) != bool(args.vision_model):
        parser.error("--reference-pairs and --vision-model must be supplied together")
    torch.set_num_threads(4)
    cached = json.loads(args.scores.read_text())
    reference = []
    reference_hash = None
    if args.reference_pairs:
        manifest_bytes = args.reference_pairs.read_bytes()
        reference_hash = hashlib.sha256(manifest_bytes).hexdigest()
        reference = list(
            score_reference_pairs(
                yaml.safe_load(manifest_bytes), args.pdfs, args.vision_model, args.device
            )
        )
    by_document = groupby(
        sorted(
            [row for row in cached["rows"] if not row["same_region"]] + reference,
            key=lambda row: row["document"],
        ),
        key=lambda row: row["document"],
    )
    rows = []
    for document, group in by_document:
        with fitz.open(args.pdfs / f"{document}.pdf") as pdf:
            lines = {}
            for row in group:
                for side in ("left", "right"):
                    page_number = row[f"{side}_page"]
                    page = pdf[page_number - 1]
                    if page_number not in lines:
                        lines[page_number] = page_lines(page)
                    row[f"{side}_source_context"] = source_context(
                        lines[page_number],
                        rect(page, row[f"{side}_bbox"]),
                        rect(page, row[f"{side}_figure_bbox"]),
                        row[f"{side}_label"],
                    )
                rows.append(row)
    vectors = embedding_lookup(
        (row[f"{side}_source_context"] for row in rows for side in ("left", "right")),
        args.model,
    )
    for row in rows:
        score = float(vectors[row["left_source_context"]] @ vectors[row["right_source_context"]])
        row["scoped_text"] = score
        for view in ("figure", "crop"):
            row[f"scoped_text_{view}"] = (score + row[view]) / 2
    development = [row for row in rows if row["split"] == "development"]
    heldout = [row for row in rows if row["split"] == "heldout"]
    summary = {}
    for method in METHODS:
        cutoffs = {
            kind: threshold(*scores_for(development, method, kind), kind)
            for kind in ("same", "different")
        }
        field = f"{method}_decision"
        for row in rows:
            same_name = _label(row["left_label"]) == _label(row["right_label"])
            kind, link = ("same", "merge") if same_name else ("different", "alias")
            row[field] = link if row[method] >= cutoffs[kind] else "none"
        summary[method] = {
            **_metrics(heldout, field),
            "development_cutoffs": cutoffs,
            "heldout_auc": {
                kind: auc(*scores_for(heldout, method, kind)) for kind in ("same", "different")
            },
        }
    args.output.write_text(
        json.dumps(
            {
                "annotation_sha256": cached["sha256"],
                "seed": cached["seed"],
                "text_model": args.model,
                "development_pairs": len(development),
                "heldout_pairs": len(heldout),
                "summary": summary,
                "reference_sha256": reference_hash,
                "reference_vision_model": args.vision_model,
                "reference_summary": {
                    method: _metrics(reference, f"{method}_decision") for method in METHODS
                }
                if reference
                else None,
                "rows": rows,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
