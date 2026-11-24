"""Replay retrieval and measure annotated one-based PDF evidence-page coverage."""

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from statistics import mean

from src.config import load_config
from src.embeddings import SiglipEncoder
from src.graph_builder import GraphBuilder
from src.layout import load_document
from src.retriever import Retriever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--matched-k", type=int, help="Use the same evidence item limit in both layers"
    )
    parser.add_argument("--render-cache", type=Path)
    parser.add_argument("--entropy-cutoff", type=float, help="Add an entropy-only routing control")
    args = parser.parse_args()
    if args.matched_k is not None and args.matched_k < 1:
        parser.error("--matched-k must be positive")
    if args.entropy_cutoff is not None and not 0 <= args.entropy_cutoff <= 1:
        parser.error("--entropy-cutoff must be in [0, 1]")
    manifest = json.loads(args.manifest.read_text())
    config = load_config(args.config)
    encoder = SiglipEncoder(config.encoder)
    builder = GraphBuilder(encoder, config.graph)
    settings = config.retrieval
    if args.matched_k is not None:
        settings = replace(settings, macro_k=args.matched_k, micro_k=args.matched_k)
    variants = {
        "hivg-rag": (settings, "adaptive"),
        "macro-only": (settings, "macro"),
        "micro-only": (settings, "micro"),
        "adaptive-with-four-parents": (replace(settings, top_parents=4), "adaptive"),
    }
    if args.entropy_cutoff is not None:
        variants["entropy-only"] = (
            replace(settings, lambda_lex=0.0, tau_zoom=args.entropy_cutoff),
            "adaptive",
        )
    rows = []
    for raw in manifest["documents"]:
        doc = load_document(
            raw, args.manifest.parent, args.render_cache or args.output.parent, config.dpi
        )
        index = builder.build(doc)
        macro_pages = {node.id: node.page + 1 for node in doc.macro}
        for method, (settings, route) in variants.items():
            retriever = Retriever(encoder, settings)
            for question in raw["questions"]:
                expected = set(
                    question.get(
                        "upstream_evidence_pdf_pages", question.get("evidence_pages_raw", [])
                    )
                )
                if not expected:
                    continue
                if not expected <= {page + 1 for page in doc.pages}:
                    raise ValueError(
                        f"{raw['id']}/{question['id']}: missing evidence pages {expected}"
                    )
                retrieved = retriever.retrieve(index, question["question"], route)
                parent_pages = {macro_pages[node] for node in retrieved.trace["selected_parents"]}
                returned = {item.page + 1 for item in retrieved.evidence if item.page is not None}
                visual = {
                    item.region.page + 1
                    for item in retrieved.evidence
                    if item.region is not None and item.include_image
                }
                rows.append(
                    {
                        "method": method,
                        "document": raw["id"],
                        "id": question["id"],
                        "subset": question.get("subset", "unlabeled"),
                        "mode": retrieved.trace["mode"],
                        "gamma": retrieved.trace["gamma"],
                        "expected": sorted(expected),
                        "parent_pages": sorted(parent_pages),
                        "predicted": sorted(returned),
                        "visual_pages": sorted(visual),
                        "evidence_items": len(retrieved.evidence),
                        "visual_items": sum(
                            item.region is not None and item.include_image
                            for item in retrieved.evidence
                        ),
                        "parent_page_recall": (
                            len(expected & parent_pages) / len(expected)
                            if route != "micro"
                            else None
                        ),
                        "page_recall": len(expected & returned) / len(expected),
                        "image_page_recall": len(expected & visual) / len(expected),
                        "all_evidence_pages": expected <= returned,
                    }
                )
        print(raw["id"], flush=True)
    summary = {}
    for method in variants:
        all_rows = [row for row in rows if row["method"] == method]
        summary[method] = {}
        for subset in ("overall", "macro", "micro"):
            group = [row for row in all_rows if subset == "overall" or row["subset"] == subset]
            parent_recalls = [
                row["parent_page_recall"] for row in group if row["parent_page_recall"] is not None
            ]
            summary[method][subset] = {
                "questions": len(group),
                "routes": dict(Counter(row["mode"] for row in group)),
                "mean_evidence_items": mean(row["evidence_items"] for row in group)
                if group
                else None,
                "mean_visual_items": mean(row["visual_items"] for row in group) if group else None,
                "mean_parent_page_recall": mean(parent_recalls) if parent_recalls else None,
                "mean_page_recall": mean(row["page_recall"] for row in group) if group else None,
                "mean_image_page_recall": (
                    mean(row["image_page_recall"] for row in group) if group else None
                ),
                "all_evidence_pages_n": sum(row["all_evidence_pages"] for row in group),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "matched_k": args.matched_k,
                "entropy_cutoff": args.entropy_cutoff,
                "summary": summary,
                "rows": rows,
            },
            indent=2,
        )
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
