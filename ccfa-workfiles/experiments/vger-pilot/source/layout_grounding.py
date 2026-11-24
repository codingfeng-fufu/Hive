"""Evaluate grounding using PDF text coordinates, captions, and image bounds."""

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import fitz

from eval.vger_pilot import _label, _overlap

MARKER = re.compile(r"^\s*(?:\(([0-9A-Za-z]{1,3})\)|(\d{1,2})[.)]?)\s*(?=\D|$)")
FIGURE = re.compile(
    r"^\s*((?:figure|fig\.?|table)\s+\d+(?:[-.]\d+)*[a-z]?)\s*[.:\-]?\s*", re.IGNORECASE
)
CAPTION_END = re.compile(r"\s+(?:is|are|with|shown|for|of|in|on|to)\b|[.:;]\s", re.IGNORECASE)


@dataclass(frozen=True)
class Candidate:
    name: str
    box: tuple[float, float, float, float]
    source: str
    distance: float


def names(text: str) -> set[str]:
    text = " ".join(text.split())
    if not text:
        return set()
    values = set()
    if marker := MARKER.match(text):
        if marker[1] or not marker[2].isdecimal() or text[marker.end() :][:1].isalpha():
            values.add(marker[1] or marker[2])
        text = text[marker.end() :]
    if figure := FIGURE.match(text):
        values.add(re.sub(r"^fig\.?\s", "Figure ", figure[1], flags=re.IGNORECASE))
        text = text[figure.end() :]
    for phrase in (text, CAPTION_END.split(text, maxsplit=1)[0]):
        if 1 <= len(phrase.split()) <= 10 and any(char.isalpha() for char in phrase):
            values.add(phrase.strip(" .:;-"))
    return {_label(value) for value in values if _label(value)}


def _box(page: fitz.Page, bbox: tuple) -> fitz.Rect:
    rect = fitz.Rect(bbox) * page.rotation_matrix
    limits = page.rect
    return fitz.Rect(
        rect.x0 * 1000 / limits.width,
        rect.y0 * 1000 / limits.height,
        rect.x1 * 1000 / limits.width,
        rect.y1 * 1000 / limits.height,
    ) & fitz.Rect(0, 0, 1000, 1000)


def _association(text: fitz.Rect, figure: fitz.Rect) -> float | None:
    horizontal = max(0, min(text.x1, figure.x1) - max(text.x0, figure.x0))
    vertical = max(0, min(text.y1, figure.y1) - max(text.y0, figure.y0))
    text_area = text.get_area()
    if text_area and horizontal * vertical / text_area >= 0.75:
        return 0.0
    gap_below = text.y0 - figure.y1
    if horizontal >= 0.65 * min(text.width, figure.width) and -8 <= gap_below <= 42:
        return 10 + abs(gap_below)
    gap_right = text.x0 - figure.x1
    if vertical >= 0.65 * min(text.height, figure.height) and -8 <= gap_right <= 115:
        return 15 + abs(gap_right)
    return None


def propose(page: fitz.Page, blocks: list[dict]) -> list[Candidate]:
    figures = []
    candidates = []
    for block in blocks:
        if block["type"] not in {"image", "table", "chart"} or not block.get("bbox"):
            continue
        box = fitz.Rect(block["bbox"])
        if box.is_empty or box.get_area() >= 850_000:
            continue
        figures.append(box)
        captions = block.get("image_caption") or block.get("table_caption") or []
        if isinstance(captions, str):
            captions = [captions]
        for caption in captions:
            for name in sorted(names(caption)):
                candidates.append(Candidate(name, tuple(box), "mineru_caption", 0.0))
    for image in page.get_image_info():
        box = _box(page, image["bbox"])
        if 800 <= box.get_area() < 850_000:
            figures.append(box)
    flags = fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES
    for block in page.get_text("dict", flags=flags)["blocks"]:
        lines = block["lines"]
        spans = [("".join(span["text"] for span in line["spans"]), line["bbox"]) for line in lines]
        if len(spans) > 1:
            spans.append((" ".join(text for text, _ in spans), block["bbox"]))
        for text, bbox in spans:
            labels = names(text)
            if not labels:
                continue
            text_box = _box(page, bbox)
            options = [
                (distance, figure)
                for figure in figures
                if (distance := _association(text_box, figure)) is not None
            ]
            options.sort(key=lambda option: (option[0], option[1].get_area(), tuple(option[1])))
            for distance, figure in options[:2]:
                for name in sorted(labels):
                    candidates.append(Candidate(name, tuple(figure), "pdf_text_geometry", distance))
    ranked = sorted(
        candidates,
        key=lambda item: (item.distance, item.source != "pdf_text_geometry", item.box, item.name),
    )
    unique = {}
    for item in ranked:
        unique.setdefault((item.name, item.box), item)
    return list(unique.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layouts", type=Path, required=True)
    parser.add_argument("--pdfs", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    predicted = {}
    for path in sorted(args.layouts.glob("*.json")):
        blocks_by_page = defaultdict(list)
        for block in json.loads(path.read_text()):
            blocks_by_page[int(block["page_idx"]) + 1].append(block)
        with fitz.open(args.pdfs / f"{path.stem}.pdf") as pdf:
            for page, blocks in sorted(blocks_by_page.items()):
                predicted[path.stem, page] = propose(pdf[page - 1], blocks)

    pairs = json.loads(args.pairs.read_text())["rows"]
    anchors = {
        (
            row["document"],
            row["split"],
            row[f"{side}_page"],
            row[f"{side}_label"],
            tuple(row[f"{side}_bbox"]),
        )
        for row in pairs
        for side in ("left", "right")
    }
    rows = []
    found = {}
    for doc, split, page, name, target in sorted(anchors):
        matches = [item for item in predicted.get((doc, page), ()) if item.name == _label(name)]
        ious = [_overlap(target, item.box) for item in matches[:5]]
        hit = max(ious, default=0.0) >= 0.5
        found[doc, page, name, target] = hit
        rows.append(
            {
                "document": doc,
                "split": split,
                "page": page,
                "name": name,
                "gold_bbox": target,
                "name_proposed": bool(matches),
                "top1_iou": max(ious[:1], default=0.0),
                "top5_iou": max(ious, default=0.0),
                "candidates": [
                    {"bbox": item.box, "source": item.source, "distance": item.distance, "iou": iou}
                    for item, iou in zip(matches[:5], ious, strict=True)
                ],
            }
        )
    pair_rows = []
    for row in pairs:
        both = all(
            found[
                row["document"],
                row[f"{side}_page"],
                row[f"{side}_label"],
                tuple(row[f"{side}_bbox"]),
            ]
            for side in ("left", "right")
        )
        pair_rows.append(
            {
                "document": row["document"],
                "split": row["split"],
                "stratum": row["stratum"],
                "same_region": row["same_region"],
                "both_endpoints_top5": both,
            }
        )
    summary = {}
    for split in ("development", "heldout", "all"):
        selected = [row for row in rows if split in {"all", row["split"]}]
        critical = [
            row
            for row in pair_rows
            if split in {"all", row["split"]}
            and not row["same_region"]
            and row["stratum"] in {"same_name_different", "different_name_same"}
        ]
        summary[split] = {
            "endpoints": len(selected),
            "names": sum(row["name_proposed"] for row in selected),
            "top1_iou_05": sum(row["top1_iou"] >= 0.5 for row in selected),
            "top5_iou_05": sum(row["top5_iou"] >= 0.5 for row in selected),
            "critical_distinct_region_pairs": len(critical),
            "both_endpoints_top5": sum(row["both_endpoints_top5"] for row in critical),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "summary": summary,
                "endpoints": rows,
                "pairs": pair_rows,
                "proposals": sum(map(len, predicted.values())),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
