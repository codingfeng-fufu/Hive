"""Diagnose which component of the QAHT score distinguishes development subsets."""

import argparse
import json
import re
from math import isclose
from pathlib import Path

from src.config import load_config
from src.retriever import lexical_prior

BARE_COUNT = re.compile(r"\bhow many\b", re.IGNORECASE)


def _auroc(rows: list[dict], field: str) -> float:
    positive = [row[field] for row in rows if row["subset"] == "micro"]
    negative = [row[field] for row in rows if row["subset"] == "macro"]
    return sum(
        0.5 if isclose(a, b, abs_tol=1e-10) else float(a > b) for a in positive for b in negative
    ) / (len(positive) * len(negative))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config).retrieval
    weight = config.lambda_lex
    if weight == 1:
        raise ValueError("Cannot recover entropy focus from a purely lexical score")
    manifest = json.loads(args.manifest.read_text())
    questions = {
        (doc["id"], q["id"]): q["question"]
        for doc in manifest["documents"]
        for q in doc["questions"]
    }
    rows = []
    for row in json.loads(args.retrieval.read_text())["rows"]:
        if row["method"] != "hivg-rag" or row["subset"] not in {"micro", "macro"}:
            continue
        question = questions[row["document"], row["id"]]
        lexical = lexical_prior(question)
        adjusted_lexical = lexical_prior(BARE_COUNT.sub("", question))
        focus = max(0.0, min(1.0, (row["gamma"] - weight * lexical) / (1 - weight)))
        rows.append(
            {
                "document": row["document"],
                "id": row["id"],
                "subset": row["subset"],
                "lexical": lexical,
                "entropy_focus": focus,
                "gamma": row["gamma"],
                "gamma_without_bare_count": weight * adjusted_lexical + (1 - weight) * focus,
            }
        )
    summary = {
        name: {"auroc": _auroc(rows, name)}
        for name in ("lexical", "entropy_focus", "gamma", "gamma_without_bare_count")
    }
    for name in ("gamma", "gamma_without_bare_count"):
        summary[name]["zoomed"] = {
            subset: sum(row["subset"] == subset and row[name] >= config.tau_zoom for row in rows)
            for subset in ("macro", "micro")
        }
    count = sum(row["gamma"] >= config.tau_zoom for row in rows)
    if 0 < count < len(rows):
        for name in ("entropy_focus", "gamma_without_bare_count"):
            ranked = sorted(rows, key=lambda row: (-row[name], row["document"], row["id"]))
            boundary = (ranked[count - 1][name], ranked[count][name])
            summary[name]["matched_zoom_count"] = {
                subset: sum(row["subset"] == subset for row in ranked[:count])
                for subset in ("macro", "micro")
            }
            if boundary[0] > boundary[1]:
                summary[name]["development_cutoff"] = sum(boundary) / 2
    args.output.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
