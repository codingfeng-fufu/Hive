"""Fetch the source images named by E-VQA's Google Landmarks IDs."""

import argparse
import csv
import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from PIL import Image


def download(image_id: str, url: str | None, output: Path) -> dict:
    destination = output / f"{image_id}.jpg"
    result = {"image_id": image_id, "source_url": url}
    if destination.is_file():
        with Image.open(destination) as image:
            image.verify()
        return {**result, "status": "cached"}
    if not url or urlsplit(url).scheme not in {"http", "https"}:
        return {**result, "status": "unmapped"}
    request = Request(url, headers={"User-Agent": "HiVG-RAG document QA research/1.0"})
    try:
        with tempfile.TemporaryDirectory(dir=output) as staging:
            image_path = Path(staging) / destination.name
            with urlopen(request, timeout=20) as response, image_path.open("wb") as stream:
                while chunk := response.read(1 << 20):
                    stream.write(chunk)
                    if stream.tell() > 50_000_000:
                        raise ValueError("Source image exceeds 50 MB")
            with Image.open(image_path) as image:
                image.verify()
            image_path.replace(destination)
    except (URLError, OSError, ValueError) as error:
        return {**result, "status": "failed", "error": str(error)}
    return {**result, "status": "downloaded"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    with args.candidates.open() as stream:
        image_ids = {
            pair["dataset_image_id"]
            for pair in map(json.loads, stream)
            if pair["dataset_name"] == "landmarks"
        }
    if not image_ids or any(
        len(image_id) != 16 or not all(c in "0123456789abcdef" for c in image_id)
        for image_id in image_ids
    ):
        raise ValueError("Expected Google Landmarks image IDs as 16 lowercase hex characters")
    with args.metadata.open(newline="", encoding="utf-8") as stream:
        urls = {row["id"]: row["url"] for row in csv.DictReader(stream) if row["id"] in image_ids}
    args.images.mkdir(parents=True, exist_ok=True)
    args.results.parent.mkdir(parents=True, exist_ok=True)

    def fetch(image_id: str) -> dict:
        return download(image_id, urls.get(image_id), args.images)

    counts: dict[str, int] = {}
    with args.results.open("w") as stream, ThreadPoolExecutor(max_workers=4) as pool:
        for row in pool.map(fetch, sorted(image_ids)):
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            status = row["status"]
            counts[status] = counts.get(status, 0) + 1
    print(json.dumps({"image_ids": len(image_ids), "mapped": len(urls), **counts}), flush=True)


if __name__ == "__main__":
    main()
