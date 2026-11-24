"""Build query-image-conditioned E-VQA manifests from a fixed Lens candidate pool."""

import argparse
import csv
import hashlib
import json
import os
import textwrap
from collections import defaultdict
from pathlib import Path

from PIL import Image


def _relative(path: Path, output: Path) -> str:
    return os.path.relpath(path.resolve(), output.parent.resolve())


def _bundle(urls: tuple[str, ...], articles: dict, media: dict, blank: str, output: Path) -> dict:
    digest = hashlib.sha256(json.dumps(urls).encode()).hexdigest()[:24]
    bundle = {
        "id": f"lens-{digest}",
        "candidate_urls": list(urls),
        "pages": {},
        "macro": [],
        "mentions": [],
        "questions": [],
    }
    for url in urls:
        article = articles[url]
        regions = defaultdict(list)
        for section, image_url, caption in zip(
            article["image_section_indices"],
            article["image_urls"],
            article["image_reference_descriptions"],
            strict=True,
        ):
            if image_path := media.get(image_url):
                regions[section].append((image_path, caption))
        for section, (title, body) in enumerate(
            zip(article["section_titles"], article["section_texts"], strict=True)
        ):
            context = f"{article['title']} · {title}"
            page = len(bundle["pages"])
            parent = f"section-{page}"
            bundle["pages"][str(page)] = blank
            bundle["macro"].append(
                {"id": parent, "kind": "section", "text": f"{context}. {body[:750]}", "page": page}
            )
            for i, span in enumerate(textwrap.wrap(body, width=550, break_long_words=False)):
                bundle["mentions"].append(
                    {
                        "id": f"{parent}:s{i}",
                        "parent": parent,
                        "kind": "span",
                        "text": f"{context}. {span}",
                        "page": page,
                    }
                )
            for path, caption in regions[section]:
                image_page = len(bundle["pages"])
                figure = f"figure-{image_page}"
                with Image.open(path) as image:
                    bbox = [0, 0, image.width, image.height]
                bundle["pages"][str(image_page)] = _relative(path, output)
                bundle["macro"].append(
                    {
                        "id": figure,
                        "kind": "figure",
                        "text": f"{context}. {caption}",
                        "page": image_page,
                        "bbox": bbox,
                        "references": [parent],
                    }
                )
                bundle["mentions"].append(
                    {
                        "id": f"{figure}:region",
                        "parent": figure,
                        "kind": "region",
                        "text": caption or context,
                        "page": image_page,
                        "bbox": bbox,
                    }
                )
    if not bundle["macro"]:
        bundle["pages"]["0"] = blank
        bundle["macro"].append(
            {
                "id": "empty-pool",
                "kind": "section",
                "text": "The candidate pool contains no article sections.",
                "page": 0,
            }
        )
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--query-images", type=Path, required=True)
    parser.add_argument(
        "--article-image-map", type=Path, help="JSON mapping image URL to local path"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--available-images-only", action="store_true")
    args = parser.parse_args()
    with args.questions.open(newline="") as stream:
        questions = list(csv.DictReader(stream))
    with (args.candidates / "articles.jsonl").open() as stream:
        articles = {article["url"]: article for article in map(json.loads, stream)}
    with (args.candidates / "query-candidates.jsonl").open() as stream:
        pairs = [json.loads(line) for line in stream]
    media = {}
    if args.article_image_map:
        media = {
            url: args.article_image_map.parent / path
            for url, path in json.loads(args.article_image_map.read_text()).items()
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    blank = args.output.with_suffix(".blank.png")
    Image.new("RGB", (28, 28), "white").save(blank)
    groups = {}
    missing = []
    for row in pairs:
        source, image_id = row["dataset_name"], row["dataset_image_id"]
        path = args.query_images / source / f"{image_id}.jpg"
        if not path.is_file():
            missing.append(row)
            continue
        urls = tuple(row["candidate_urls"])
        if urls not in groups:
            groups[urls] = _bundle(
                urls, articles, media, _relative(blank, args.output), args.output
            )
        question = questions[int(row["question_id"])]
        groups[urls]["questions"].append(
            {
                "id": f"{row['question_id']}-{image_id}",
                "question": question["question"],
                "answers": [question["answer"]],
                "image": _relative(path, args.output),
                "image_source": source,
                "image_id": image_id,
                "gold_in_candidate_pool": row["gold_in_candidates"],
            }
        )
    coverage = {
        "query_image_pairs": len(pairs),
        "missing_query_image_pairs": len(missing),
        "missing_image_ids": sorted(
            {f"{row['dataset_name']}/{row['dataset_image_id']}" for row in missing}
        ),
        "documents": len(groups),
        "article_image_urls": len(media),
    }
    args.output.with_suffix(".coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
    if missing and not args.available_images_only:
        raise SystemExit(f"Query images missing for {len(missing)} of {len(pairs)} pairs")
    if not groups:
        raise SystemExit("No cases with query images")
    manifest = {
        "dataset": "e-vqa-lens",
        "split": "test" if not missing else "test-available-images",
        "candidate_protocol": "official-lens-top-k",
        "documents": list(groups.values()),
    }
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: value for key, value in coverage.items() if key != "missing_image_ids"}))


if __name__ == "__main__":
    main()
