"""Check if a fixed ColPali ranking supports the chart intervention."""

import argparse
from pathlib import Path

import pymupdf
from PIL import Image, ImageOps
from probe import is_correct, query_chart, render, shifted_labels, swapped, top_chart

from eval.baselines import ColPali
from eval.colpali_granularity import page_layout, read, save
from src.layout import load_document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rank-cache", type=Path)
    parser.add_argument(
        "--size", nargs=2, type=int, default=(336, 504), metavar=("WIDTH", "HEIGHT")
    )
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("cases.json"))
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument("--edit-mode", choices=("swap", "shift"), default="swap")
    parser.add_argument(
        "--views",
        nargs="+",
        choices=("page", "layout-context", "query-fixed", "query-adaptive"),
        default=("page", "layout-context"),
    )
    parser.add_argument("--second-page", choices=("retrieve", "blank"), default="retrieve")
    args = parser.parse_args()
    size = tuple(args.size)
    if any(edge < 56 or edge % 28 for edge in size):
        parser.error("--size must use multiples of 28 of at least 56 pixels")
    budget = 2 * (size[0] // 28) * (size[1] // 28)
    args.output.mkdir(parents=True, exist_ok=True)
    case = next(c for c in read(args.cases) if c["anchor"] == args.anchor)
    documents = read(Path("data/mmlongbench-full.json"))["documents"]
    raw = next(d for d in documents if any(q["id"] == args.anchor for q in d["questions"]))
    ranks_path = args.rank_cache or args.output / "rankings.json"
    if ranks_path.exists():
        ranks = read(ranks_path)
    else:
        model = ColPali()
        doc = load_document(raw, Path("data"), Path("data/rendered"), 250, require_graph=False)
        print(f"Indexing {len(doc.pages)} pages", flush=True)
        with model.torch.autocast("cpu", dtype=model.torch.bfloat16):
            pages, embeddings = model.index(doc)
        index = pages, [embedding.float() for embedding in embeddings]
        ranks = {}
        for q, _, _ in case["questions"]:
            result = model.retrieve(index, q)
            ranks[q] = [{"page": item.page + 1, "score": item.score} for item in result.evidence]
        save(ranks_path, ranks)
        print(ranks, flush=True)
        del model, index, embeddings
    if args.index_only:
        return

    from eval.granularity_reader import Reader

    reader = Reader(size)
    rows = []
    blocks, source = page_layout(raw)
    with pymupdf.open(Path("data") / raw["pdf"]) as pdf:
        gold = render(pdf, case)
        if args.edit_mode == "shift":
            edited = shifted_labels(gold, case["swap"])
            edited.save(args.output / "shifted-page.png")
        else:
            edited = swapped(gold, *case["swap"])
        for question, original_value, edited_value in case["questions"]:
            if args.edit_mode == "shift":
                edited_value = original_value
            count = 2 if args.second_page == "retrieve" else 1
            candidates = [row["page"] for row in ranks[question][:count]]
            selections = {}
            for view in args.views:
                selected = []
                for page in candidates:
                    bbox = (0, 0, *gold.size)
                    details = {"kind": "page"}
                    if view == "layout-context":
                        try:
                            bbox, _ = top_chart(raw, page, gold.size, include_context=True)
                            details = {"kind": "chart"}
                        except ValueError:
                            pass
                    if view in {"query-fixed", "query-adaptive"}:
                        bbox, details = query_chart(
                            pdf[page - 1],
                            blocks[page - 1],
                            question,
                            gold.size,
                            adapt_extent=view == "query-adaptive",
                        )
                    selected.append({"bbox": bbox, "page": page, **details})
                selections[view] = selected
            for condition, expected in (("original", original_value), ("edited", edited_value)):
                native = [
                    (gold if condition == "original" else edited)
                    if page == case["page"]
                    else render(pdf, {**case, "page": page})
                    for page in candidates
                ]
                for view, selected in selections.items():
                    images = [
                        ImageOps.pad(image.crop(selection["bbox"]), size, color="white")
                        for image, selection in zip(native, selected, strict=True)
                    ]
                    if args.second_page == "blank":
                        images.append(Image.new("RGB", size, "white"))
                    result = reader.answer(question + " Return a number only.", images)
                    assert result["visual_tokens"] == budget
                    row = {
                        "question": question,
                        "condition": condition,
                        "edit_mode": args.edit_mode,
                        "view": view,
                        "image_size": size,
                        "second_page": args.second_page,
                        "pages": candidates,
                        "selected": selected,
                        "layout": source,
                        "gold_page": case["page"],
                        "expected": expected,
                        "correct": is_correct(result["answer"], expected),
                        **result,
                    }
                    rows.append(row)
                    print(view, condition, candidates, result["answer"], expected, flush=True)
    save(args.output / "results.json", rows)


if __name__ == "__main__":
    main()
