"""Index each PDF once and cache page rankings for paired QA."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

from eval.baselines import ColPali, VisRAG
from src.layout import load_document
from src.retriever import Evidence, Retrieval


class CachedPageRetriever:
    method = "visrag"

    def __init__(self) -> None:
        cache = Path(os.getenv("HIVG_PAGE_BASELINE_CACHE", "runs/vger-pilot"))
        path = cache / f"{self.method}-rankings.json"
        self.data = json.loads(path.read_text())

    def index(self, document):
        return self.data[document.id]

    def retrieve(self, index, question):
        stored = index["questions"][question]
        return Retrieval(
            [Evidence(**row) for row in stored["evidence"]],
            {
                "mode": self.method,
                "precomputed_index_seconds": index["index_seconds"],
                "precomputed_retrieval_seconds": stored["retrieval_seconds"],
            },
        )


class CachedVisRAG(CachedPageRetriever):
    method = "visrag"


class CachedColPali(CachedPageRetriever):
    method = "colpali"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("visrag", "colpali"), default="visrag")
    parser.add_argument("--resume", action="store_true", help="Reuse completed document rankings")
    args = parser.parse_args()
    method = {"visrag": VisRAG, "colpali": ColPali}[args.method]()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = json.loads(args.output.read_text()) if args.resume and args.output.exists() else {}
    seen = set()
    for path in args.manifest:
        manifest = json.loads(path.read_text())
        for raw in manifest["documents"]:
            if raw["id"] in seen:
                raise ValueError(f"Duplicate document ID {raw['id']!r}")
            seen.add(raw["id"])
            if raw["id"] in output:
                if set(output[raw["id"]]["questions"]) != {q["question"] for q in raw["questions"]}:
                    raise ValueError(f"Cached questions changed for document {raw['id']!r}")
                continue
            document = load_document(
                raw, path.parent, Path("data/rendered"), 250, require_graph=False
            )
            started = perf_counter()
            index = method.index(document)
            elapsed = perf_counter() - started
            rows = {}
            for q in raw["questions"]:
                query_started = perf_counter()
                predictions = method.retrieve(index, q["question"])
                rows[q["question"]] = {
                    "evidence": [vars(item) for item in predictions.evidence],
                    "retrieval_seconds": perf_counter() - query_started,
                }
            output[document.id] = {"index_seconds": elapsed, "questions": rows}
            staging = args.output.with_suffix(".tmp")
            staging.write_text(json.dumps(output, indent=2))
            staging.replace(args.output)
            print(manifest["dataset"], raw["id"], len(rows), round(elapsed, 2), flush=True)


if __name__ == "__main__":
    main()
