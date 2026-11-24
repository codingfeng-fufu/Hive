"""Test graph-level VGER when entity names and boxes are supplied by annotation."""

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import fitz
import yaml

from eval.metrics import topology
from eval.vger_pilot import _crop, load_pairs
from src.config import load_config
from src.document import Document, MacroNode, Mention
from src.embeddings import SiglipEncoder
from src.graph_builder import GraphBuilder, predicted_links
from src.layout import native_region


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--pdfs", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alias-page-windows", nargs="+", type=int, default=[])
    args = parser.parse_args()
    if any(value < 0 for value in args.alias_page_windows):
        parser.error("--alias-page-windows must contain nonnegative values")
    annotations = yaml.safe_load(args.annotations.read_text())
    config = load_config(args.config)
    encoder = SiglipEncoder(config.encoder)
    indices, rows = [], []
    for name, (split, pairs) in load_pairs(annotations, seed=42).items():
        occurrences = sorted(
            {item for _, a, b in pairs for item in (a, b)},
            key=lambda item: (item.page, item.bbox, item.label),
        )
        ids = {item: f"mention-{i}" for i, item in enumerate(occurrences)}
        gold = {key: item.entity for item, key in ids.items()}
        figures = annotations["documents"][name]["figures"]
        macros = tuple(
            MacroNode(f"figure-{key}", "figure", caption, page - 1)
            for key, (page, _, caption) in figures.items()
        )
        pdf_path = args.pdfs / f"{name}.pdf"
        with fitz.open(pdf_path) as pdf:
            mentions = tuple(
                Mention(
                    ids[item],
                    f"figure-{item.figure}",
                    item.label,
                    item.page - 1,
                    native_region(
                        item.page - 1,
                        list(item.bbox),
                        (
                            round(pdf[item.page - 1].rect.width * config.dpi / 72),
                            round(pdf[item.page - 1].rect.height * config.dpi / 72),
                        ),
                        "normalized_1000",
                    ),
                )
                for item in occurrences
            )
            all_boxes = sorted(
                {(item.page, box) for item in occurrences for box in (item.bbox, item.figure_bbox)}
            )
            vectors = encoder.image([_crop(pdf, page, box) for page, box in all_boxes])
        region_vectors = dict(zip(all_boxes, vectors, strict=True))
        nodes = (*macros, *mentions)
        text = dict(zip((node.id for node in nodes), encoder.text([node.text for node in nodes])))
        document = Document(name, {}, macros, mentions, pdf_path)
        variants = [
            ("text", "text", replace(config.graph, enable_split=False, enable_bridge=False)),
            ("figure", "figure", config.graph),
            ("crop", "crop", config.graph),
        ]
        variants.extend(
            (f"crop-window-{window}", "crop", replace(config.graph, alias_page_window=window))
            for window in dict.fromkeys(args.alias_page_windows)
        )
        for variant, representation, graph in variants:
            visual = (
                {}
                if representation == "text"
                else {
                    ids[item]: region_vectors[
                        item.page, item.figure_bbox if representation == "figure" else item.bbox
                    ]
                    for item in occurrences
                }
            )
            index = GraphBuilder(encoder, graph).build(document, features=(text, visual))
            indices.append(
                {
                    "document": name,
                    "split": split,
                    "variant": variant,
                    **topology(index, gold, graph.superhub_degree),
                }
            )
            merged, alias = predicted_links(index, set(gold))
            for stratum, left, right in pairs:
                pair = tuple(sorted((ids[left], ids[right])))
                rows.append(
                    {
                        "document": name,
                        "split": split,
                        "variant": variant,
                        "stratum": stratum,
                        "left": ids[left],
                        "right": ids[right],
                        "left_label": left.label,
                        "right_label": right.label,
                        "page_gap": abs(left.page - right.page),
                        "same_region": (left.page, left.bbox) == (right.page, right.bbox),
                        "merged": pair in merged,
                        "alias": pair in alias,
                    }
                )
        print(name, len(occurrences), flush=True)
    counts = Counter()
    for row in rows:
        key = row["split"], row["variant"], row["stratum"]
        counts[(*key, "n")] += 1
        counts[(*key, "merged")] += row["merged"]
        counts[(*key, "alias")] += row["alias"]
    summary = {"/".join(key): count for key, count in sorted(counts.items())}
    args.output.write_text(
        json.dumps({"summary": summary, "documents": indices, "pairs": rows}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
