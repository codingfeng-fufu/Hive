"""Evaluate fixed and query-dependent chart crops on unchanged benchmark questions."""

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean

import pymupdf
from PIL import Image, ImageOps
from probe import query_chart, top_chart

from eval.colpali_granularity import HELDOUT, page_layout, read, save, words
from eval.granularity_reader import Reader
from eval.metrics import score
from eval.official import mmlongbench

SIZE = (336, 504)
THUMBNAIL = (252, 336)
METHODS = ("page", "layout-context", "query-fixed", "query-adaptive")
PAGE_BUDGETS = {
    "page": (SIZE, SIZE),
    "first-large": ((392, 672), (224, 336)),
    "second-large": ((224, 336), (392, 672)),
}


def page_context(page, query):
    """Return the most relevant PDF text blocks in their original reading order."""
    terms = words(query)
    blocks = [
        " ".join(block[4].split())
        for block in page.get_text("blocks", sort=True)
        if block[6] == 0 and block[4].strip()
    ]
    matched = sorted(
        (i for i, block in enumerate(blocks) if terms & words(block)),
        key=lambda i: (-len(terms & words(blocks[i])), i),
    )
    return "\n".join(blocks[i] for i in sorted(matched[:3]))[:800]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument("--page-budget", action="store_true")
    comparison.add_argument("--context-pair", action="store_true")
    comparison.add_argument("--text-context", action="store_true")
    parser.add_argument(
        "--rankings", type=Path, default=Path("runs/colpali-granularity/heldout-rankings.json")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rankings = read(args.rankings)
    prefix, methods = "benchmark", METHODS
    if args.page_budget:
        prefix, methods = "page-budget", tuple(PAGE_BUDGETS)
    elif args.context_pair:
        prefix, methods = "context-pair", ("page", "context-duplicate", "context-crop")
    elif args.text_context:
        prefix, methods = "text-context", ("page-text", "crop-text")
    results = args.output / f"{prefix}-results.json"
    rows = read(results) if results.exists() else []
    completed = {(r["document"], r["id"], r["method"]) for r in rows}
    reader = Reader(
        (392, 672) if args.page_budget else SIZE,
        variable_size=args.page_budget or args.context_pair,
    )
    for doc in read(HELDOUT)["documents"]:
        blocks, _ = page_layout(doc)
        with pymupdf.open(Path("data") / doc["pdf"]) as pdf:
            pages = {}
            for question in doc["questions"]:
                query = question["question"]
                ranks = rankings[doc["id"]]["questions"][query]["evidence"]
                selected_pages = [entry["page"] for entry in ranks[:2]]
                assert len(selected_pages) == 2
                decodes = {}
                evidence = None
                if args.text_context:
                    context = [page_context(pdf[p], query) for p in selected_pages]
                    evidence = (
                        "\n\n".join(
                            f"Page {p + 1}: {content}"
                            for p, content in zip(selected_pages, context, strict=True)
                            if content
                        )
                        or None
                    )
                for method in methods:
                    key = (doc["id"], question["id"], method)
                    if key in completed:
                        continue
                    images, boxes, image_pages, sizes = [], [], [], []
                    page_sizes = PAGE_BUDGETS[method] if args.page_budget else (SIZE, SIZE)
                    for number, size in zip(selected_pages, page_sizes, strict=True):
                        page = pdf[number]
                        if number not in pages:
                            pix = page.get_pixmap(dpi=115, alpha=False)
                            pages[number] = Image.frombytes(
                                "RGB", (pix.width, pix.height), pix.samples
                            )
                        image = pages[number]
                        box = (0, 0, *image.size)
                        if method in {"context-duplicate", "context-crop"}:
                            size = THUMBNAIL
                            boxes.append(box)
                            sizes.append(size)
                            image_pages.append(number + 1)
                            images.append(ImageOps.pad(image, size, color="white"))
                        if method == "layout-context":
                            try:
                                box, _ = top_chart(
                                    doc, number + 1, image.size, include_context=True
                                )
                            except ValueError:
                                pass
                        if method in {"query-fixed", "query-adaptive", "context-crop", "crop-text"}:
                            box, _ = query_chart(
                                page,
                                blocks[number],
                                query,
                                image.size,
                                adapt_extent=method == "query-adaptive",
                            )
                        boxes.append(box)
                        sizes.append(size)
                        image_pages.append(number + 1)
                        images.append(ImageOps.pad(image.crop(box), size, color="white"))
                    input_key = tuple(zip(boxes, sizes, strict=True))
                    if input_key not in decodes:
                        decodes[input_key] = reader.answer(query, images, evidence)
                    output = decodes[input_key]
                    assert output["visual_tokens"] == 432
                    answer = output["answer"]
                    rows.append(
                        {
                            "document": doc["id"],
                            "id": question["id"],
                            "question": query,
                            "method": method,
                            "pages": [p + 1 for p in selected_pages],
                            "bboxes": boxes,
                            "image_sizes": sizes,
                            "image_pages": image_pages,
                            "evidence_text": evidence,
                            "chart_table_figure": bool(
                                set(question["evidence_sources"]) & {"Chart", "Table", "Figure"}
                            ),
                            **output,
                            **score(question, answer),
                            "official_score": mmlongbench(question, answer),
                        }
                    )
                    save(results, rows)
                    completed.add(key)
        print(doc["id"], len(rows), flush=True)
    groups = defaultdict(list)
    for row in rows:
        groups[row["method"], "all"].append(row)
        key = "chart_table_figure" if row["chart_table_figure"] else "other"
        groups[row["method"], key].append(row)
    summary = {
        f"{method}/{subset}": {
            "n": len(group),
            **{
                metric: mean(r[metric] for r in group)
                for metric in (
                    "accuracy",
                    "anls",
                    "official_score",
                    "visual_tokens",
                    "prompt_tokens",
                )
            },
        }
        for (method, subset), group in groups.items()
    }
    save(args.output / f"{prefix}-summary.json", summary)


if __name__ == "__main__":
    main()
