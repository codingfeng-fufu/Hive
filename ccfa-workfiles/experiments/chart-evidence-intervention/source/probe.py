"""Evaluate numeric readout before and after a local label exchange."""

import argparse
import math
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path

import pymupdf
from PIL import Image, ImageOps

from eval.colpali_granularity import SIZE, page_layout, pdf_rect, read, save, words


def render(document, case):
    page = document[case["page"] - 1]
    pixmap = page.get_pixmap(dpi=115, alpha=False)
    assert (pixmap.width, pixmap.height) == tuple(case["reference_size"])
    return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


def swapped(image, first, second):
    restore = image.copy()
    top, bottom = image.crop(first), image.crop(second)
    if top.size != bottom.size:
        raise ValueError("Labels must occupy equally sized rectangles")
    restore.paste(bottom, first)
    restore.paste(top, second)
    assert restore.tobytes() != image.tobytes()
    return restore


def shifted_labels(image, boxes, offset=3):
    """Translate labels inside their rectangles without changing their assignments."""
    edited = image.copy()
    for box in boxes:
        patch = image.crop(box)
        width, height = patch.size
        leading = patch.crop((0, 0, 1, height))
        trailing_margin = patch.crop((width - offset, 0, width, height))
        if min(low for low, _ in trailing_margin.getextrema()) < 240:
            offset_for_box = -offset
            edge = patch.crop((0, 0, offset, height))
            if min(low for low, _ in edge.getextrema()) < 240:
                raise ValueError(f"No clear margin for label shift in {box}")
            leading = patch.crop((width - 1, 0, width, height))
        else:
            offset_for_box = offset
        moved = leading.resize(patch.size)
        moved.paste(patch, (offset_for_box, 0))
        if moved.tobytes() == patch.tobytes():
            raise ValueError(f"Shifting the rectangle did not change its pixels: {box}")
        edited.paste(moved, box)
    return edited


def is_correct(answer, value):
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*%?\s*\.?", answer)
    return bool(match and Decimal(match[1]) == Decimal(value))


def top_chart(document, page, size, *, include_context=False):
    blocks, source = page_layout(document)
    boxes = [
        item["bbox"]
        for item in blocks.get(page - 1, [])
        if item["type"] in {"image", "table"} and item.get("bbox")
    ]
    if not boxes:
        raise ValueError(f"No automatic chart candidate on page {page} of {document['id']}")
    first = min(boxes, key=lambda box: (box[1], box[0]))
    aligned = [
        box
        for box in boxes
        if min(box[3], first[3]) - max(box[1], first[1])
        >= 0.5 * min(box[3] - box[1], first[3] - first[1])
    ]
    bounds = (
        min(x[0] for x in aligned),
        min(x[1] for x in aligned),
        max(x[2] for x in aligned),
        max(x[3] for x in aligned),
    )
    return to_pixels(expand(bounds, 70 if include_context else 20), size), source


def expand(box, above):
    return tuple(
        max(0, min(1000, value + margin))
        for value, margin in zip(box, (-20, -above, 20, 20), strict=True)
    )


def to_pixels(box, size):
    return tuple(round(value * size[i % 2] / 1000) for i, value in enumerate(box))


def query_chart(page, blocks, question, size, *, adapt_extent):
    """Select a row of chart blocks using index-time text and preserve adjacent qualifiers."""
    items = [item for item in blocks if item["type"] in {"image", "table"} and item.get("bbox")]
    items.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    remaining = list(items)
    groups = []
    while remaining:
        first = remaining.pop(0)
        start, end = first["bbox"][1::2]
        aligned = [first]
        separate = []
        for item in remaining:
            top, bottom = item["bbox"][1::2]
            target = (
                aligned
                if min(bottom, end) - max(top, start) >= 0.5 * min(end - start, bottom - top)
                else separate
            )
            target.append(item)
        remaining = separate
        rects = [item["bbox"] for item in aligned]
        box = (
            min(rect[0] for rect in rects),
            min(rect[1] for rect in rects),
            max(rect[2] for rect in rects),
            max(rect[3] for rect in rects),
        )
        caption = " ".join(
            part
            for item in aligned
            for field in ("image_caption", "table_caption")
            for part in item.get(field, [])
        )
        local = words(page.get_text(clip=pdf_rect(page, expand(box, 20))))
        wide = words(page.get_text(clip=pdf_rect(page, expand(box, 70))))
        groups.append((box, words(caption) | wide, wide - local))
    if not groups:
        return (0, 0, *size), {"kind": "page", "matched": []}

    terms = {term for term in words(question) if term[0].isalpha()}
    frequency = Counter(term for _, evidence, _ in groups for term in evidence & terms)
    weights = {
        term: 1 + math.log((len(groups) + 1) / (count + 1)) for term, count in frequency.items()
    }
    box, evidence, above = max(
        groups, key=lambda group: sum(weights[term] for term in sorted(group[1] & terms))
    )
    matched = evidence & terms
    if not matched:
        return (0, 0, *size), {"kind": "page", "matched": []}
    pad = 20 if adapt_extent and not (above & terms) else 70
    return to_pixels(expand(box, pad), size), {
        "kind": "chart",
        "matched": sorted(matched),
        "upper_context": sorted(above & terms),
        "top_padding": pad,
        "groups": len(groups),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("cases.json"))
    parser.add_argument("--edit-mode", choices=("swap", "shift"), default="swap")
    parser.add_argument("--size", nargs=2, type=int, default=SIZE, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument(
        "--views",
        nargs="+",
        choices=("page", "figure", "layout", "layout-context"),
        default=("page", "figure"),
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    size = tuple(args.size)
    if any(edge < 56 or edge % 28 for edge in size):
        parser.error("--size must use multiples of 28 of at least 56 pixels")
    budget = (size[0] // 28) * (size[1] // 28)
    target = args.output
    (target / "images").mkdir(parents=True, exist_ok=True)
    data = Path("data")
    documents = read(data / "mmlongbench-full.json")["documents"]
    anchors = {
        question["id"]: (doc, question["evidence_pages_raw"])
        for doc in documents
        for question in doc["questions"]
    }
    cases = read(args.cases)
    path = target / "results.json"
    rows = read(path) if path.is_file() else []
    if any(row["visual_tokens"] != budget or tuple(row.get("size", SIZE)) != size for row in rows):
        parser.error("Existing results use a different image size")
    if any(row.get("edit_mode", "swap") != args.edit_mode for row in rows):
        parser.error("Existing results use a different edit mode")
    completed = {(row["anchor"], row["view"], row["condition"], row["query_id"]) for row in rows}
    if not args.prepare_only:
        from eval.granularity_reader import Reader

        reader = Reader(size)
    for case in cases:
        anchor = case["anchor"]
        document_record, gold_pages = anchors[anchor]
        relative_pdf = document_record["pdf"]
        assert gold_pages == [case["page"]]
        with pymupdf.open(data / relative_pdf) as document:
            before = render(document, case)
        if args.edit_mode == "shift":
            after = shifted_labels(before, case["swap"])
        else:
            after = swapped(before, *case["swap"])
        layout_source = None
        boxes = {"page": (0, 0, *before.size), "figure": case["figure"]}
        for view in set(args.views) & {"layout", "layout-context"}:
            boxes[view], layout_source = top_chart(
                document_record, case["page"], before.size, include_context=view == "layout-context"
            )
        for condition, native in (("original", before), ("edited", after)):
            for view in args.views:
                image = ImageOps.pad(native.crop(boxes[view]), size, color="white")
                image_path = target / "images" / f"{anchor}-{condition}-{view}.png"
                image.save(image_path)
                if args.prepare_only:
                    continue
                for query_id, (question, expected_before, expected_after) in enumerate(
                    case["questions"]
                ):
                    key = (anchor, view, condition, query_id)
                    if key in completed:
                        continue
                    expected = expected_before
                    if condition == "edited" and args.edit_mode == "swap":
                        expected = expected_after
                    result = reader.answer(question + " Return a number only.", [image])
                    assert result["visual_tokens"] == budget
                    rows.append(
                        {
                            "anchor": anchor,
                            "pdf": relative_pdf,
                            "page": case["page"],
                            "condition": condition,
                            "edit_mode": args.edit_mode,
                            "view": view,
                            "crop_bbox": boxes[view],
                            "layout": layout_source if view.startswith("layout") else None,
                            "size": size,
                            "query_id": query_id,
                            "question": question,
                            "expected": expected,
                            "affected": expected_before != expected_after,
                            "correct": is_correct(result["answer"], expected),
                            "image": str(image_path),
                            **result,
                        }
                    )
                    save(path, rows)
                    completed.add(key)
        print(anchor, len(rows), flush=True)
    if args.prepare_only:
        return
    for view in args.views:
        selected = [r for r in rows if r["view"] == view]
        assert len(selected) == sum(len(case["questions"]) * 2 for case in cases)
        by_key = {(r["anchor"], r["query_id"], r["condition"]): r for r in selected}
        for affected in (True, False):
            originals = [
                r for r in selected if r["condition"] == "original" and r["affected"] == affected
            ]
            edited = [by_key[(r["anchor"], r["query_id"], "edited")] for r in originals]
            print(
                view,
                affected,
                "original",
                sum(r["correct"] for r in originals),
                "edited",
                sum(r["correct"] for r in edited),
                "paired",
                sum(a["correct"] and b["correct"] for a, b in zip(originals, edited, strict=True)),
                "n",
                len(originals),
                flush=True,
            )


if __name__ == "__main__":
    main()
