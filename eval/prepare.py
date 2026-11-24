import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from src.layout import parse_pdf


def _relative(path: Path, destination: Path) -> str:
    return os.path.relpath(path.resolve(), destination.parent.resolve())


def _answer_list(value: object, *, list_answer: bool = False) -> list[str]:
    if isinstance(value, list) and not list_answer:
        return [str(answer) for answer in value]
    return [str(value)]


def _pdf_entry(pdf: Path, document_id: str, options: argparse.Namespace) -> dict:
    if not pdf.is_file():
        raise FileNotFoundError(pdf)
    entry = {"id": document_id, "pdf": _relative(pdf, options.output), "questions": []}
    if options.parse:
        layout = parse_pdf(pdf, options.layouts, options.mineru_backend)
    elif options.layouts:
        layouts = sorted(options.layouts.rglob(f"{pdf.stem}_content_list.json"))
        if len(layouts) != 1:
            raise ValueError(f"Expected one layout for {pdf}, found {len(layouts)}")
        layout = layouts[0]
    else:
        return entry
    entry["layout"] = _relative(layout, options.output)
    return entry


def _dataset(source: Path):
    if source.suffix == ".parquet" or source.is_dir():
        from datasets import load_dataset

        files = [str(source)] if source.is_file() else sorted(map(str, source.glob("*.parquet")))
        if not files:
            raise ValueError(f"No Parquet files in {source}")
        return load_dataset("parquet", data_files=files, split="train")
    if source.suffix == ".jsonl":
        return [json.loads(line) for line in source.read_text().splitlines() if line]
    return json.loads(source.read_text())


def _mmlongbench(options: argparse.Namespace) -> list[dict]:
    entries = {}
    for i, row in enumerate(_dataset(options.source)):
        doc_id = row["doc_id"]
        if options.doc and doc_id not in options.doc:
            continue
        if doc_id not in entries:
            entries[doc_id] = _pdf_entry(options.assets / doc_id, doc_id, options)
        entries[doc_id]["questions"].append(
            {
                "id": str(i),
                "question": row["question"],
                "answers": [str(row["answer"])],
                "answer_format": row["answer_format"],
                "doc_type": row["doc_type"],
                "evidence_pages_raw": ast.literal_eval(row["evidence_pages"]),
                "evidence_sources": ast.literal_eval(row["evidence_sources"]),
            }
        )
    return list(entries.values())


def _docbench(options: argparse.Namespace) -> list[dict]:
    entries = []
    for path in sorted(options.source.rglob("*_qa.jsonl")):
        doc_id = path.name.removesuffix("_qa.jsonl")
        if options.doc and doc_id not in options.doc:
            continue
        matches = sorted((options.assets / doc_id).glob("*.pdf"))
        pdf = matches[0] if len(matches) == 1 else options.assets / f"{doc_id}.pdf"
        entry = _pdf_entry(pdf, doc_id, options)
        for i, row in enumerate(_dataset(path)):
            entry["questions"].append(
                {
                    "id": str(row.get("id", i)),
                    "question": row["question"],
                    "answers": _answer_list(row["answer"]),
                    "upstream": row,
                }
            )
        entries.append(entry)
    return entries


def _slidevqa(options: argparse.Namespace) -> list[dict]:
    entries = {}
    for row in _dataset(options.source):
        doc_id = row["deck_name"]
        if options.doc and doc_id not in options.doc:
            continue
        if doc_id not in entries:
            # Prefer an original PDF when one is available. Otherwise build a PDF for MinerU
            # while preserving the released slide raster as the source for native crops.
            pdf = options.assets / f"{doc_id}.pdf"
            if pdf.is_file():
                entries[doc_id] = _pdf_entry(pdf, doc_id, options)
            else:
                import fitz

                key = hashlib.sha256(doc_id.encode()).hexdigest()[:24]
                stage = options.assets / "slidevqa-extracted" / key
                stage.mkdir(parents=True, exist_ok=True)
                images = []
                with fitz.open() as deck:
                    if row.get("image_urls"):
                        # The official SlideVQA downloader uses the original URL basenames.
                        slides = [
                            options.assets / doc_id / Path(urlsplit(url).path).name
                            for url in row["image_urls"]
                        ]
                    else:
                        slides = []
                        for i in range(1, 21):
                            image = row.get(f"page_{i}")
                            if image is None:
                                break
                            path = stage / f"{i:02}.png"
                            image.save(path, "PNG")
                            slides.append(path)
                    for path in slides:
                        with fitz.open(path) as raster:
                            width, height = raster[0].rect.width, raster[0].rect.height
                        images.append(path)
                        page = deck.new_page(width=width, height=height)
                        page.insert_image(page.rect, filename=path)
                    if not images:
                        raise ValueError(f"{doc_id}: no slide assets")
                    pdf = stage / f"{key}.pdf"
                    deck.save(pdf)
                entries[doc_id] = _pdf_entry(pdf, doc_id, options)
                entries[doc_id]["pages"] = [_relative(path, options.output) for path in images]
        entries[doc_id]["questions"].append(
            {
                "id": str(row["qa_id"]),
                "question": row["question"],
                "answers": [str(row["answer"])],
                "evidence_pages_raw": row["evidence_pages"],
                "arithmetic_expression": row.get("arithmetic_expression"),
            }
        )
    return list(entries.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare benchmark assets and stable question IDs")
    parser.add_argument("dataset", choices=("mmlongbench-doc", "docbench", "slidevqa", "e-vqa"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--doc", action="append", help="select a document by source ID")
    parser.add_argument("--layouts", type=Path, help="directory with MinerU 2.x content_list files")
    parser.add_argument(
        "--parse", action="store_true", help="generate MinerU 2.x content_list files"
    )
    parser.add_argument("--mineru-backend", default="pipeline")
    parser.add_argument("--annotations", type=Path, help="JSON keyed by document ID")
    options = parser.parse_args()
    if options.parse and options.layouts is None:
        parser.error("--parse requires --layouts")
    converters = {
        "mmlongbench-doc": _mmlongbench,
        "docbench": _docbench,
        "slidevqa": _slidevqa,
    }
    if options.dataset == "e-vqa":
        # E-VQA can use several evidence corpora. Require the fixed candidate bundles as input.
        docs = json.loads(options.source.read_text())["documents"]
        docs = [doc for doc in docs if not options.doc or doc["id"] in options.doc]
        for doc in docs:
            for item in doc["questions"]:
                if item.get("image"):
                    item["image"] = _relative(options.assets / item["image"], options.output)
            for name in ("pdf", "layout"):
                if name in doc:
                    doc[name] = _relative(options.assets / doc[name], options.output)
            if "pages" in doc:
                if isinstance(doc["pages"], dict):
                    doc["pages"] = {
                        key: _relative(options.assets / path, options.output)
                        for key, path in doc["pages"].items()
                    }
                else:
                    doc["pages"] = [
                        _relative(options.assets / path, options.output) for path in doc["pages"]
                    ]
            if options.parse and "layout" not in doc and "pdf" in doc:
                doc["layout"] = _relative(
                    parse_pdf(
                        options.output.parent / doc["pdf"],
                        options.layouts,
                        options.mineru_backend,
                    ),
                    options.output,
                )
    else:
        docs = converters[options.dataset](options)
    if not docs:
        parser.error("No documents matched the selection")
    if options.annotations:
        labels = json.loads(options.annotations.read_text())
        for doc in docs:
            annotation = labels.get(doc["id"], {})
            for field in ("entity_gold", "distractors"):
                if field in annotation:
                    doc[field] = annotation[field]
            for item in doc.get("distractors", []):
                item["image"] = _relative(
                    options.annotations.parent / item["image"], options.output
                )
            for item in doc["questions"]:
                item["subset"] = annotation.get("subsets", {}).get(item["id"], "unlabeled")
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(
            {
                "dataset": options.dataset,
                "split": options.split,
                "documents": docs,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
