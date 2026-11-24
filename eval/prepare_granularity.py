"""Assign conservative evidence-based granularity labels for a first-pass analysis."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

VISUAL_TYPES = {"Chart", "Figure", "Table"}
LOCAL_VISUAL = re.compile(
    r"\b(?:table|figure|fig\.|chart|graph)\b.{0,180}\b(?:cell|row|column|axis|point|bar|legend)\b"
    r"|\b(?:cell|row|column|axis|point|bar|legend)\b.{0,180}\b(?:table|figure|chart|graph)\b",
    re.IGNORECASE | re.DOTALL,
)
MULTI_SOURCE = re.compile(
    r"\b(?:across|throughout|all)\s+(?:the\s+)?(?:pages|chapters|sections|figures|tables)\b"
    r"|\b(?:figure|table|section)\s*\d+(?:\.\d+)*\s+and\s+(?:figure|table|section)\s*\d+",
    re.IGNORECASE,
)


def classify(dataset: str, question: dict) -> str:
    if dataset == "mmlongbench-doc":
        pages = set(question["evidence_pages_raw"])
        sources = set(question["evidence_sources"])
        if len(pages) >= 3:
            return "macro"
        if len(pages) == 1 and len(sources) == 1 and sources <= VISUAL_TYPES:
            return "micro"
    elif dataset == "docbench":
        upstream = question["upstream"]
        source = upstream.get("evidence") or ""
        if upstream["type"] in {"text-only", "multimodal-f", "multimodal-t"}:
            if MULTI_SOURCE.search(source):
                return "macro"
            if upstream["type"] in {"multimodal-f", "multimodal-t"} and LOCAL_VISUAL.search(source):
                return "micro"
    return "unlabeled"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reviewed", type=Path, help="Manifest of manually reviewed subset IDs")
    args = parser.parse_args()
    if args.manifest.parent.resolve() != args.output.parent.resolve():
        parser.error("Write to the manifest directory to preserve relative asset paths")
    if args.manifest.resolve() == args.output.resolve():
        parser.error("Use a separate output for provisional labels")
    data = json.loads(args.manifest.read_text())
    reviewed = {}
    if args.reviewed:
        source = json.loads(args.reviewed.read_text())
        reviewed = {
            (document["id"], question["id"]): question["subset"]
            for document in source["documents"]
            for question in document["questions"]
            if "subset" in question
        }
    counts: Counter[str] = Counter()
    for document in data["documents"]:
        for question in document["questions"]:
            key = document["id"], question["id"]
            if "subset" not in question:
                question["subset"] = (
                    reviewed[key] if key in reviewed else classify(data["dataset"], question)
                )
            counts[question["subset"]] += 1
    data["granularity_protocol"] = "provisional-evidence-v1"
    args.output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(data["dataset"], dict(counts))


if __name__ == "__main__":
    main()
