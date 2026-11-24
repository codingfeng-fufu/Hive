"""Combine QA runs, replacing failed rows only after a successful exact-case retry."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from eval.metrics import summarize


def _paired(left: dict, right: dict, metric: str) -> dict | None:
    by_doc = defaultdict(list)
    for (doc, q), item in left.items():
        baseline = right.get((doc, q), {})
        a, b = item.get(metric), baseline.get(metric)
        if item["status"] == baseline.get("status") == "ok" and a is not None and b is not None:
            by_doc[doc].append(a - b)
    if not by_doc:
        return None
    rng = random.Random(42)
    docs = list(by_doc)
    bootstrap = []
    for _ in range(3000):
        resample = rng.choices(docs, k=len(docs))
        bootstrap.append(mean(value for doc in resample for value in by_doc[doc]))
    return {
        "paired_documents": len(docs),
        "paired_questions": sum(map(len, by_doc.values())),
        "mean_delta": mean(value for group in by_doc.values() for value in group),
        "document_bootstrap_95ci": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
    }


def _comparison(rows: dict, baseline: dict) -> dict:
    return {
        subset: _paired(
            {
                case: row
                for case, row in rows.items()
                if subset == "overall" or row["subset"] == subset
            },
            baseline,
            "anls",
        )
        for subset in ("overall", "macro", "micro")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument(
        "--replace-document",
        nargs=5,
        action="append",
        default=[],
        metavar=("DATASET", "MODEL", "METHOD", "SETTING", "DOCUMENT"),
        help="Replace a complete document from the final --run, including prior successful rows",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = defaultdict(dict)
    replacements = {tuple(values) for values in args.replace_document}
    replacements_seen = defaultdict(set)
    for root in args.run:
        for path in sorted(root.rglob("predictions.jsonl")):
            dataset, model, method, setting, _ = path.relative_to(root).parts
            key = dataset, model, method, setting
            for line in path.read_text().splitlines():
                row = json.loads(line)
                case = row["doc_id"], row["question_id"]
                previous = groups[key].get(case)
                replacing = (*key, case[0]) in replacements and root == args.run[-1]
                if replacing:
                    if previous is None or row["status"] != "ok":
                        raise ValueError(f"Unmatched or unsuccessful replacement for {key}/{case}")
                    if any(
                        previous[field] != row[field] for field in ("question", "answers", "subset")
                    ):
                        raise ValueError(f"Replacement case content changed for {key}/{case}")
                    replacements_seen[(*key, case[0])].add(case)
                elif previous and previous["status"] == "ok":
                    raise ValueError(f"Duplicate successful prediction for {key}/{case}")
                groups[key][case] = row
    for *key, document in replacements:
        expected = {case for case in groups[tuple(key)] if case[0] == document}
        if not expected or replacements_seen[(*key, document)] != expected:
            raise ValueError(f"Replacement did not include all questions for {key}/{document}")
    outputs = []
    for (dataset, model, method, setting), rows in sorted(groups.items()):
        summary = summarize(list(rows.values()), [])
        baseline = groups[dataset, model, "direct", setting]
        outputs.append(
            {
                "dataset": dataset,
                "model": model,
                "method": method,
                "setting": setting,
                "subsets": {
                    key: summary[key] for key in ("overall", "macro", "micro", "unlabeled")
                },
                "vs_direct": _paired(rows, baseline, "anls") if method != "direct" else None,
                **(
                    {
                        "vs_fixed_routes": {
                            route: _comparison(rows, groups[dataset, model, route, setting])
                            for route in ("macro-only", "micro-only")
                            if (dataset, model, route, setting) in groups
                        }
                    }
                    if method == "hivg-rag"
                    else {}
                ),
            }
        )
    args.output.write_text(json.dumps(outputs, indent=2))
    for item in outputs:
        overall = item["subsets"]["overall"]
        print(
            item["dataset"],
            item["model"],
            item["method"],
            {
                k: overall[k]
                for k in ("scored", "errors", "anls", "official_score", "prompt_tokens")
            },
        )


if __name__ == "__main__":
    main()
