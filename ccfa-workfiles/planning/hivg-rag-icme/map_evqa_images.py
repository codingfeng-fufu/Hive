"""Resolve E-VQA source image IDs against official release metadata."""

import argparse
import csv
import json
import tarfile
from collections import Counter
from pathlib import Path

import ijson


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--inat", type=Path, required=True)
    parser.add_argument("--landmarks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.candidates.open() as stream:
        pairs = [json.loads(line) for line in stream]
    image_ids = {(row["dataset_name"], row["dataset_image_id"]) for row in pairs}
    matches = {}
    with tarfile.open(args.inat, "r|gz") as archive:
        for item in archive:
            if Path(item.name).name == "train.json" and item.isfile():
                with archive.extractfile(item) as stream:
                    for image in ijson.items(stream, "images.item"):
                        key = "inaturalist", str(image["id"])
                        if key in image_ids:
                            matches[key] = {"file_name": image["file_name"]}
    with args.landmarks.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = "landmarks", row["id"]
            if key in image_ids:
                matches[key] = {"url": row["url"]}

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "query-image-sources.jsonl").open("w") as stream:
        for dataset, image_id in sorted(image_ids):
            row = {
                "dataset_name": dataset,
                "dataset_image_id": image_id,
                "source": matches.get((dataset, image_id)),
            }
            stream.write(json.dumps(row) + "\n")
    summary = {
        "unique_ids": dict(Counter(source for source, _ in image_ids)),
        "mapped_ids": dict(Counter(source for source, _ in matches)),
        "mapped_question_image_pairs": sum(
            (row["dataset_name"], row["dataset_image_id"]) in matches for row in pairs
        ),
    }
    (args.output / "query-image-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
