"""Extract the official Lens candidate articles from an E-VQA knowledge-base ZIP."""

import argparse
import ast
import csv
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit
from zipfile import ZipFile


def _key(url: str) -> tuple[str, str]:
    parsed = urlsplit(url)
    return parsed.netloc.lower(), unquote(parsed.path).replace(" ", "_")


def main() -> None:
    import ijson

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--lens", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be positive")

    with args.lens.open(newline="") as stream:
        lens = {
            (row["dataset_name"], row["dataset_image_id"]): list(
                dict.fromkeys(ast.literal_eval(row["lens_wiki_urls"]))
            )[: args.k]
            for row in csv.DictReader(stream)
        }
    with args.questions.open(newline="") as stream:
        questions = list(csv.DictReader(stream))
    wanted = {_key(url) for urls in lens.values() for url in urls}
    articles = {}
    with ZipFile(args.archive) as archive, archive.open("encyclopedic_kb_wiki.json") as stream:
        for url, article in ijson.kvitems(stream, "", use_float=True):
            key = _key(url)
            if key in wanted:
                articles[key] = article
    output = []
    for i, question in enumerate(questions):
        source = question["dataset_name"]
        gold_key = _key(question["wikipedia_url"])
        for image_id in question["dataset_image_ids"].split("|"):
            image_id = image_id.strip()
            if not image_id:
                continue
            candidates = lens.get((source, image_id), [])
            matched = [articles[_key(url)]["url"] for url in candidates if _key(url) in articles]
            output.append(
                {
                    "question_id": str(i),
                    "dataset_name": source,
                    "dataset_image_id": image_id,
                    "candidate_urls": matched,
                    "lens_url_count": len(candidates),
                    "gold_in_candidates": any(_key(url) == gold_key for url in matched),
                }
            )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "articles.jsonl").open("w") as stream:
        for article in sorted(articles.values(), key=lambda item: item["url"]):
            stream.write(json.dumps(article, ensure_ascii=False) + "\n")
    with (args.output / "query-candidates.jsonl").open("w") as stream:
        for row in output:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "k": args.k,
        "questions": len(questions),
        "question_image_pairs": len(output),
        "unique_lens_urls": len(wanted),
        "matched_articles": len(articles),
        "pairs_with_candidates": sum(bool(row["candidate_urls"]) for row in output),
        "pairs_with_gold_candidate": sum(row["gold_in_candidates"] for row in output),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
