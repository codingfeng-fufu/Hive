"""Parse bounded page ranges and assemble a complete MinerU document."""

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import fitz


def read_chunk(path: Path, stem: str, count: int) -> list[dict]:
    middle = json.loads((path / f"{stem}_middle.json").read_text())
    if [row["page_idx"] for row in middle["pdf_info"]] != list(range(count)):
        raise ValueError(f"Incomplete page range in {path}")
    blocks = json.loads((path / f"{stem}_content_list.json").read_text())
    for block in blocks:
        if not 0 <= block["page_idx"] < count:
            raise ValueError(f"Out-of-range block in {path}")
        asset = block.get("img_path")
        if asset and not (path / asset).is_file():
            raise FileNotFoundError(path / asset)
    return blocks


def assemble(stem: str, chunks: list[tuple[int, int, Path]], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    marker = target / f"{stem}_content_list.json"
    target.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as staging:
        staged = Path(staging)
        blocks = []
        with (staged / f"{stem}.md").open("w") as markdown:
            for offset, count, path in chunks:
                items = read_chunk(path, stem, count)
                markdown.write((path / f"{stem}.md").read_text())
                markdown.write("\n\n")
                if (path / "images").is_dir():
                    shutil.copytree(path / "images", staged / "images", dirs_exist_ok=True)
                blocks.extend({**item, "page_idx": item["page_idx"] + offset} for item in items)
        if not blocks:
            raise ValueError(f"No usable blocks in {stem}")
        if (staged / "images").is_dir():
            shutil.copytree(staged / "images", target / "images", dirs_exist_ok=True)
        (staged / f"{stem}.md").replace(target / f"{stem}.md")
        content = staged / marker.name
        content.write_text(json.dumps(blocks, ensure_ascii=False, indent=2) + "\n")
        content.replace(marker)


def parse_document(pdf: Path, root: Path, url: str, chunk_pages: int) -> None:
    from mineru.cli.common import do_parse
    from pypdfium2 import PdfDocument, PdfiumError

    stem = pdf.stem
    original = pdf.read_bytes()
    was_rebuilt = False
    with fitz.open(pdf) as source:
        total = len(source)
        try:
            with PdfDocument(original):
                data = original
        except PdfiumError:
            with fitz.open() as clean:
                clean.insert_pdf(source)
                data = clean.tobytes(garbage=3, deflate=True)
            with PdfDocument(data) as repaired:
                if len(repaired) != total:
                    raise ValueError(f"PDF repair changed the page count for {pdf}")
            was_rebuilt = True
            print(f"Rebuilt PDF structure for {pdf.name} ({total} pages)", flush=True)
    digest = hashlib.sha256(original).hexdigest()[:20]
    scratch = root / ".chunks" / f"{stem}-{digest}-{chunk_pages}"
    chunks = []
    for start in range(0, total, chunk_pages):
        stop = min(total, start + chunk_pages)
        output = scratch / f"{start:05}-{stop:05}"
        path = output / stem / "vlm"
        try:
            read_chunk(path, stem, stop - start)
            if not (path / f"{stem}.md").is_file():
                raise FileNotFoundError(path / f"{stem}.md")
        except (OSError, ValueError, KeyError):
            print(f"{stem} pages {start + 1}–{stop}/{total}", flush=True)
            do_parse(
                str(output),
                [stem],
                [data],
                ["en"],
                backend="vlm-http-client",
                server_url=url,
                f_draw_layout_bbox=False,
                f_draw_span_bbox=False,
                f_dump_orig_pdf=False,
                f_dump_model_output=False,
                start_page_id=start,
                end_page_id=stop - 1,
            )
            read_chunk(path, stem, stop - start)
        chunks.append((start, stop - start, path))
    target = root / stem / "vlm"
    if was_rebuilt:
        rebuilt = target / "source" / pdf.name
        rebuilt.parent.mkdir(parents=True, exist_ok=True)
        rebuilt.write_bytes(data)
    assemble(stem, chunks, target)
    shutil.rmtree(scratch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("root", type=Path)
    parser.add_argument("url")
    parser.add_argument("--chunk-pages", type=int, default=48)
    args = parser.parse_args()
    if args.chunk_pages < 1:
        parser.error("--chunk-pages must be positive")
    parse_document(args.pdf, args.root, args.url, args.chunk_pages)
