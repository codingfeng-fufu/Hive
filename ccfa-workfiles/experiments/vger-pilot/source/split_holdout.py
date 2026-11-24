"""Compare split ablations on fixed automatic mentions from new real PDF pages."""

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import fitz
import yaml

from eval.vger_pilot import _label, _overlap
from src.config import load_config
from src.document import Mention
from src.embeddings import SiglipEncoder
from src.graph_builder import GraphBuilder
from src.layout import native_region, read_mineru
from src.retriever import Retriever

from .layout_grounding import propose


def prepare(name, pdf_path, layout, selection, cache, dpi):
    source = json.loads(layout.read_text())
    page_blocks = defaultdict(list)
    for block in source:
        if block["page_idx"] + 1 in selection:
            page_blocks[block["page_idx"] + 1].append(block)
    document_dir = cache / name
    document_dir.mkdir(parents=True, exist_ok=True)
    layout_copy = document_dir / "layout.json"
    layout_copy.write_text(json.dumps([b for p in selection for b in page_blocks[p]]))
    pages, sizes, proposals = {}, {}, {}
    with fitz.open(pdf_path) as pdf:
        for page in selection:
            source_page = pdf[page - 1]
            raster = source_page.get_pixmap(dpi=dpi, alpha=False)
            pages[page - 1] = document_dir / f"{page}.png"
            raster.save(pages[page - 1])
            sizes[page - 1] = (raster.width, raster.height)
            proposals[page - 1] = propose(source_page, page_blocks[page])
    document = read_mineru(name, pages, layout_copy, pdf_path)
    extra = []
    for page, items in proposals.items():
        parents = [node for node in document.macro if node.page == page and node.region]
        kept = defaultdict(list)
        for item in items:
            region = native_region(page, list(item.box), sizes[page], "normalized_1000")
            if any(_overlap(region.bbox, bbox) >= 0.9 for bbox in kept[item.name]):
                continue
            parent = max(
                parents,
                key=lambda node: _overlap(region.bbox, node.region.bbox),
                default=None,
            )
            if parent is None or _overlap(region.bbox, parent.region.bbox) == 0:
                continue
            kept[item.name].append(region.bbox)
            extra.append(Mention(f"geometry:{len(extra)}", parent.id, item.name, page, region))
    return document, replace(document, mentions=(*document.mentions, *extra)), sizes


def matching_mentions(document, gold):
    return {
        key: [
            mention.id
            for mention in document.mentions
            if mention.kind == "entity"
            and mention.region is not None
            and _label(mention.text) == _label(name)
            and mention.region.page == region.page
            and _overlap(mention.region.bbox, region.bbox) >= 0.5
        ]
        for key, (name, _, region) in gold.items()
    }


def pair_decisions(index, pairs, gold, matches):
    decisions = []
    for left, right in pairs:
        same = gold[left][1] == gold[right][1]
        # An automatic region that matches both anchors cannot certify two localized mentions.
        distinct = [(a, b) for a in matches[left] for b in matches[right] if a != b]
        decisions.append(
            {
                "left": left,
                "right": right,
                "same_identity": same,
                "automatic_both": bool(distinct),
                "merged": (
                    any(index.entity_of[a] == index.entity_of[b] for a, b in distinct)
                    if distinct
                    else None
                ),
            }
        )
    return decisions


def retrieval_decisions(index, queries, gold, retriever):
    results = []
    for anchor, question in queries:
        label, identity, target = gold[anchor]
        distractors = [
            region
            for other_label, other_id, region in gold.values()
            if _label(other_label) == _label(label) and other_id != identity
        ]
        evidence = retriever.retrieve(index, question, route="micro").evidence
        images = [item.region for item in evidence if item.region and item.include_image]

        def hit(region, images=images):
            return any(
                item.page == region.page and _overlap(item.bbox, region.bbox) >= 0.5
                for item in images
            )

        results.append(
            {
                "anchor": anchor,
                "query": question,
                "target_crop": hit(target),
                "has_same_name_distractor": bool(distractors),
                "wrong_crop": any(hit(region) for region in distractors),
                "selected": [
                    {
                        "occurrence_id": item.occurrence_id,
                        "node_id": item.node_id,
                        "page": item.page,
                        "bbox": item.region.bbox if item.region else None,
                        "include_image": item.include_image,
                    }
                    for item in evidence
                ],
            }
        )
    return results


def summarize(records):
    result = {}
    for family in ("production", "geometry"):
        for mode in ("text", "full-figure", "crop"):
            selected = [r for r in records if (r["family"], r["mode"]) == (family, mode)]
            pairs = [p for r in selected for p in r["pairs"]]
            negatives = [p for p in pairs if not p["same_identity"]]
            positives = [p for p in pairs if p["same_identity"]]
            queries = [q for r in selected for q in r["queries"]]
            result[f"{family}/{mode}"] = {
                "documents": len(selected),
                "negative_total": len(negatives),
                "negative_grounded_both": sum(p["automatic_both"] for p in negatives),
                "false_merges": sum(p["merged"] is True for p in negatives),
                "positive_total": len(positives),
                "positive_grounded_both": sum(p["automatic_both"] for p in positives),
                "same_identity_merges": sum(p["merged"] is True for p in positives),
                "queries": len(queries),
                "target_crop_queries": sum(q["target_crop"] for q in queries),
                "queries_with_distractor": sum(q["has_same_name_distractor"] for q in queries),
                "wrong_crop_queries": sum(q["wrong_crop"] for q in queries),
                "endpoints": sum(len(r["matches"]) for r in selected),
                "grounded_endpoints": sum(
                    sum(bool(v) for v in r["matches"].values()) for r in selected
                ),
            }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--pdfs", type=Path, required=True)
    parser.add_argument("--layouts", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    settings = load_config(args.config)
    encoder = SiglipEncoder(settings.encoder)
    graph_config = replace(settings.graph, enable_bridge=False)
    retriever = Retriever(encoder, replace(settings.retrieval, micro_k=6))
    manifest = args.annotations.read_bytes()
    records = []
    for name, raw in yaml.safe_load(manifest)["documents"].items():
        selection = sorted({entry[0] for entry in raw["occurrences"].values()})
        layout = args.layouts / name / "vlm" / f"{name}_content_list.json"
        document, extended, sizes = prepare(
            name, args.pdfs / f"{name}.pdf", layout, selection, args.output / "pages", settings.dpi
        )
        gold = {
            key: (label, entity, native_region(page - 1, box, sizes[page - 1], "normalized_1000"))
            for key, (page, box, label, entity) in raw["occurrences"].items()
        }
        for a, b in raw["pairs"]:
            if _label(gold[a][0]) != _label(gold[b][0]):
                raise ValueError(f"Pair has different names: {name}/{a}/{b}")
        builder = GraphBuilder(encoder, graph_config)
        crop_index = builder.build(extended)
        for family, current in (("production", document), ("geometry", extended)):
            matches = matching_mentions(current, gold)
            parent_of = {mention.id: mention.parent for mention in current.mentions}
            figure_visual = crop_index.visual.copy()
            for mention in current.mentions:
                if (
                    mention.kind == "entity"
                    and mention.id in figure_visual
                    and (parent := parent_of[mention.id]) in crop_index.visual
                ):
                    figure_visual[mention.id] = crop_index.visual[parent]
            for mode in ("text", "full-figure", "crop"):
                visual = figure_visual if mode == "full-figure" else crop_index.visual
                index = GraphBuilder(
                    encoder, replace(graph_config, enable_split=mode != "text")
                ).build(current, features=(crop_index.text, visual))
                # Keep retrieval similarity and visual context identical across split ablations.
                index.visual = crop_index.visual
                records.append(
                    {
                        "document": name,
                        "pages": selection,
                        "family": family,
                        "mode": mode,
                        "matches": matches,
                        "pairs": pair_decisions(index, raw["pairs"], gold, matches),
                        "queries": retrieval_decisions(index, raw["queries"], gold, retriever),
                    }
                )
        print(name, selection, flush=True)
    result = {
        "annotation_sha256": hashlib.sha256(manifest).hexdigest(),
        "summary": summarize(records),
        "records": records,
    }
    (args.output / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
