"""Compare scoped visual selection on fixed ColPali candidates."""

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter

import pymupdf as fitz
from PIL import Image, ImageOps

from eval.baselines import ColPali
from eval.metrics import score
from src.layout import load_document

ROOT = Path("data")
TARGETS = Path("ccfa-workfiles/experiments/colpali-granularity/source/targets.json")
HELDOUT = TARGETS.with_name("heldout.json")
PAGE_PILOT = ROOT / "mmlongbench-full-document-pilot.json"
CACHED_RANKS = Path("runs/vger-pilot/colpali-rankings.json")
SIZE = (504, 672)
VISUAL_TOKENS = (SIZE[0] // 28) * (SIZE[1] // 28)
TERMS = re.compile(r"[a-z]+(?:[/-][a-z]+)*|\d+(?:\.\d+)?")
STOP = {
    "a",
    "the",
    "is",
    "on",
    "of",
    "to",
    "in",
    "at",
    "an",
    "for",
    "as",
    "with",
    "which",
    "what",
    "how",
    "do",
    "does",
    "did",
    "from",
    "there",
    "and",
    "by",
    "that",
    "or",
    "are",
    "be",
    "please",
    "answer",
    "format",
    "write",
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, payload):
    stage = path.with_suffix(".tmp")
    stage.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    stage.replace(path)


def index(output, cohort):
    if cohort not in {"target", "heldout"}:
        raise ValueError("Public rankings are stored in the existing ColPali cache")
    manifest = TARGETS if cohort == "target" else HELDOUT
    path = output / f"{cohort}-rankings.json"
    ranks = read(path) if path.exists() else {}
    model = ColPali()
    fast_cpu = model.device == "cpu" and os.getenv("HIVG_COLPALI_CPU_BF16") == "1"
    autocast = (
        model.torch.autocast("cpu", dtype=model.torch.bfloat16) if fast_cpu else nullcontext()
    )
    for raw in read(manifest)["documents"]:
        if raw["id"] in ranks and set(ranks[raw["id"]]["questions"]) == {
            question["question"] for question in raw["questions"]
        }:
            continue
        doc = load_document(raw, ROOT, ROOT / "rendered", 250, require_graph=False)
        start = perf_counter()
        print(f"Indexing {raw['id']} ({len(doc.pages)} pages)", flush=True)
        with autocast:
            indexed = model.index(doc)
        seconds = perf_counter() - start
        # Float inputs keep CPU MaxSim scoring on the baseline's dtype.
        if fast_cpu:
            indexed = indexed[0], [feature.float() for feature in indexed[1]]
        questions = {}
        for q in raw["questions"]:
            start = perf_counter()
            result = model.retrieve(indexed, q["question"])
            questions[q["question"]] = {
                "evidence": [vars(item) for item in result.evidence],
                "retrieval_seconds": perf_counter() - start,
            }
        ranks[raw["id"]] = {
            "index_seconds": seconds,
            "image_encoding": "cpu-autocast-bf16" if fast_cpu else "checkpoint-default",
            "questions": questions,
        }
        save(path, ranks)
        print(raw["id"], len(doc.pages), round(seconds, 2), flush=True)


def words(text):
    return set(TERMS.findall(text.casefold())) - STOP


@dataclass(frozen=True)
class View:
    id: str
    page: int
    bbox: tuple[int, int, int, int]
    parent: str
    kind: str
    text: str
    context: str
    rank: int
    heading: str = ""


def pdf_rect(page, box):
    return fitz.Rect(
        box[0] * page.rect.width / 1000,
        box[1] * page.rect.height / 1000,
        box[2] * page.rect.width / 1000,
        box[3] * page.rect.height / 1000,
    )


def page_layout(raw):
    stem = Path(raw["pdf"]).stem
    files = sorted((ROOT / "mineru-full" / stem).glob(f"*/{stem}_content_list.json"))
    if not files:
        files = sorted((ROOT / "vger-screen/full-docs" / stem).rglob("*_content_list.json"))
    result = defaultdict(list)
    if files:
        for item in read(files[0]):
            result[item["page_idx"]].append(item)
    return result, str(files[0]) if files else None


def make_views(pdf, pages, layout):
    views = []
    for rank, number in enumerate(pages):
        page = pdf[number]
        full = (0, 0, 1000, 1000)
        page_id = f"p{number + 1}"
        views.append(View(page_id, number, full, page_id, "page", page.get_text(), "", rank))
        for i, item in enumerate(layout[number]):
            if item["type"] not in {"image", "table"} or not item.get("bbox"):
                continue
            box = tuple(max(0, min(1000, n)) for n in item["bbox"])
            x0, y0, x1, y1 = box
            if min(x1 - x0, y1 - y0) < 45:
                continue
            context = " ".join(
                str(part)
                for key in ("image_caption", "table_caption")
                for part in (item.get(key) or [])
            )
            parent = f"{page_id}b{i}"
            text = page.get_text(clip=pdf_rect(page, box))
            views.append(View(parent, number, box, parent, "block", text, context, rank))
            # Offset slices let the selector disambiguate panels while retaining the full vertical axis.
            if x1 - x0 >= 240:
                mid = (x0 + x1) // 2
                halo = (x1 - x0) // 24
                for suffix, span in (("l", (x0, mid + halo)), ("r", (mid - halo, x1))):
                    child = (span[0], y0, span[1], y1)
                    child_text = page.get_text(clip=pdf_rect(page, child))
                    views.append(
                        View(
                            parent + suffix,
                            number,
                            child,
                            parent,
                            "panel",
                            child_text,
                            context,
                            rank,
                        )
                    )
                    heading_box = (child[0], max(0, y0 - 80), child[2], y0)
                    if heading_box[1] == y0:
                        continue
                    heading_text = page.get_text(clip=pdf_rect(page, heading_box))
                    # The caption identifies text belonging to the figure, while
                    # its word positions distinguish the corresponding panels.
                    heading_terms = words(heading_text) & words(context)
                    if heading_terms:
                        expanded = (child[0], heading_box[1], child[2], child[3])
                        views.append(
                            View(
                                parent + suffix + "h",
                                number,
                                expanded,
                                parent,
                                "panel-heading",
                                child_text,
                                context,
                                rank,
                                " ".join(sorted(heading_terms)),
                            )
                        )
    return views


def choose(question, views, method):
    page_views = [v for v in views if v.kind == "page"]
    if method == "colpali":
        return page_views[:2]
    if method == "colpali-top6":
        return page_views[:6]
    if method == "text-layout":
        method = "page-scope"
    if method == "heading-text":
        method = "heading-scope"
    query = words(question)
    with_headings = method in {"heading-relevance", "heading-scope"}
    grouped = [
        v for v in views if v.kind != "page" and (with_headings or v.kind != "panel-heading")
    ] or page_views
    fields = {view.id: words(view.text) for view in grouped}
    contexts = {view.id: words(view.context) for view in grouped}
    frequency = Counter(word for field in fields.values() for word in field)
    weight = {word: 1 + math.log((len(grouped) + 1) / (frequency[word] + 1)) for word in query}

    def mass(terms):
        return sum(weight[word] for word in sorted(terms & query))

    def relevance(view):
        return mass(fields[view.id]) + 0.55 * mass(contexts[view.id]) + 2 / (1 + view.rank)

    ordered = sorted(grouped, key=lambda view: (-relevance(view), view.rank, view.id))
    if method in {"page-relevance", "page-scope", "heading-relevance", "heading-scope"}:
        selected = []
        for page in page_views[:2]:
            blocks = [view for view in ordered if view.page == page.page and view.kind == "block"]
            if not blocks:
                selected.append(page)
                continue
            block = blocks[0]
            block_support = (fields[block.id] | contexts[block.id]) & query
            page_support = words(page.text) & query
            if len(block_support) < 2 or mass(page_support - block_support) > mass(block_support):
                selected.append(page)
                continue
            if method == "page-scope":
                children = [
                    view for view in ordered if view.parent == block.id and view.kind == "panel"
                ]
                if len(children) == 2:
                    child, sibling = children
                    child_terms = fields[child.id] & query
                    sibling_terms = fields[sibling.id] & query
                    # Repeated evidence offers two interpretations. Prefer a panel
                    # only when the discriminating terms favor it unambiguously.
                    if (
                        child_terms & sibling_terms
                        and mass(child_terms - sibling_terms) > mass(sibling_terms - child_terms)
                        and mass(block_support - contexts[child.id] - child_terms)
                        <= mass(child_terms)
                    ):
                        block = child
            if with_headings:
                children = [
                    v for v in ordered if v.parent == block.id and v.kind == "panel-heading"
                ]
                if len(children) == 2:
                    first, second = children
                    first_heading = words(first.heading) - words(second.heading)
                    second_heading = words(second.heading) - words(first.heading)
                    if method == "heading-relevance":
                        first_score = mass(fields[first.id] | words(first.heading))
                        second_score = mass(fields[second.id] | words(second.heading))
                    else:
                        first_score = (
                            mass(first_heading)
                            if fields[first.id] & fields[second.id] & query
                            else 0
                        )
                        second_score = (
                            mass(second_heading)
                            if fields[first.id] & fields[second.id] & query
                            else 0
                        )
                    if first_score != second_score:
                        child = first if first_score > second_score else second
                        if method == "heading-scope" or mass(words(child.heading)) > 0:
                            block = child
            selected.append(block)
        return selected
    # Use the same candidate views in both selection rules.
    chosen = []
    for view in ordered:
        if len(chosen) == 2:
            break
        if not chosen or view.parent != chosen[0].parent:
            chosen.append(view)
    if method == "relevance":
        return (chosen + [v for v in page_views if v.page not in {x.page for x in chosen}])[:2]

    viable = ordered[:12]
    distinction = {}
    for view in viable:
        terms = fields[view.id]
        scope = set()
        for peer in viable:
            # Shared query terms identify a plausible competing reading. Distinguishing
            # terms must be localized to the view and absent from that competitor.
            if (
                peer.id != view.id
                and peer.kind == view.kind
                and (terms & fields[peer.id] & query or peer.parent == view.parent)
            ):
                scope.update((terms - fields[peer.id]) & query)
        distinction[view.id] = mass(scope)
    first = max(viable, key=lambda v: (relevance(v) + distinction[v.id], -v.rank))
    # If the higher resolution selection is a panel, retain its owning block as
    # context. If a full block suffices, use the second view for a new page.
    if first.kind == "panel":
        parent = next(v for v in grouped if v.id == first.parent)
        residual = query - fields[first.id] - contexts[first.id]
        alternative = next((v for v in ordered if v.page != first.page), None)
        if alternative is None or not (
            (fields[alternative.id] | contexts[alternative.id]) & residual
        ):
            return [first, parent]
    second = next((v for v in ordered if v.page != first.page), None)
    if second is None:
        second = next((v for v in page_views if v.page != first.page), None)
    return [first, second] if second else [first]


def coverage(predicted, page, target):
    x0, y0, x1, y1 = target
    area = (x1 - x0) * (y1 - y0)
    return any(
        v.page + 1 == page
        and max(0, min(x1, v.bbox[2]) - max(x0, v.bbox[0]))
        * max(0, min(y1, v.bbox[3]) - max(y0, v.bbox[1]))
        >= 0.8 * area
        for v in predicted
    )


def render(pdf, view):
    page = pdf[view.page]
    rgb = page.get_pixmap(dpi=144, clip=pdf_rect(page, view.bbox), alpha=False)
    image = Image.frombytes("RGB", (rgb.width, rgb.height), rgb.samples)
    return ImageOps.pad(image, SIZE, color="white")


def text_evidence(pdf, selected):
    evidence = []
    for view in selected:
        page = pdf[view.page]
        rect = pdf_rect(page, view.bbox)
        spans = []
        for block in page.get_text("dict", clip=rect)["blocks"]:
            for line in block.get("lines", []):
                x, y, _, _ = line["bbox"]
                text = " ".join(span["text"] for span in line["spans"]).strip()
                if text:
                    spans.append(f"x={round(x)} y={round(y)} {text}")
        evidence.append(
            f"Page {view.page + 1}, crop {view.bbox}.\n"
            + (f"Caption: {view.context}\n" if view.context else "")
            + "\n".join(spans)[:6500]
        )
    return "\n\n".join(evidence)


def score_answer(question, answer):
    if question.get("pair") and re.fullmatch(r"\s*\d+(?:\.\d+)?\s*%?\s*", answer):
        # Every answer-labeled probe asks for a percentage on the 0–100 scale.
        answer = answer.strip().removesuffix("%").strip()
        question = {**question, "answers": [v.removesuffix("%") for v in question["answers"]]}
    return score(question, answer)


METHODS = ("colpali", "relevance", "scope")
ALL_METHODS = (
    *METHODS,
    "page-relevance",
    "page-scope",
    "colpali-top6",
    "text-layout",
    "heading-relevance",
    "heading-scope",
    "heading-text",
    "verification-control",
    "verification",
)


def verification_views(pdf, question, views, reader):
    defaults = choose(question, views, "heading-scope")
    probes = []
    for view in defaults:
        siblings = [
            candidate
            for candidate in views
            if candidate.parent == view.parent and candidate.kind == "panel-heading"
        ]
        probes.extend(siblings if siblings else [view])
    if len(probes) < 4:
        probes.extend(view for view in views if view.kind == "page" and view.rank == 2)
    probes = probes[:4]
    preview = Image.new("RGB", SIZE, "white")
    from PIL import ImageDraw

    draw = ImageDraw.Draw(preview)
    width, height = SIZE[0] // 2, SIZE[1] // 2
    for i, view in enumerate(probes):
        x, y = (i % 2) * width, (i // 2) * height
        thumb = ImageOps.contain(render(pdf, view), (width - 8, height - 26))
        preview.paste(thumb, (x + (width - thumb.width) // 2, y + 24))
        draw.text((x + 6, y + 4), str(i + 1), fill="black")
    observation = reader.answer(
        question,
        [preview],
        instruction=(
            "Compare the numbered thumbnails. Check whether the visible chart title, legend, "
            "or table heading uniquely supports the subject and qualifier of the question. "
            "Select at most one thumbnail. If several may match or none are legible, use 0. "
            "Return JSON only with integer choice and a short literal clue visible in the image. "
            'Do not compute or give the answer. Format: {"choice":0,"clue":""}.'
        ),
    )
    match = re.search(r"\{.*?\}", observation["answer"], re.DOTALL)
    try:
        decision = json.loads(match[0]) if match is not None else {}
    except json.JSONDecodeError:
        decision = {}
    choice = decision.get("choice")
    clue = decision.get("clue")
    refined = defaults.copy()
    if type(choice) is int and 1 <= choice <= len(probes) and isinstance(clue, str) and words(clue):
        candidate = probes[choice - 1]
        slot = next(
            (i for i, view in enumerate(defaults) if view.page == candidate.page),
            len(defaults) - 1,
        )
        refined[slot] = candidate
    return (
        defaults,
        refined,
        {
            **observation,
            "choice": choice,
            "clue": clue,
            "preview_regions": [
                {"id": view.id, "page": view.page + 1, "bbox": view.bbox} for view in probes
            ],
        },
    )


def experiment(output, cohort, retrieval_only, methods):
    manifest = {"public": PAGE_PILOT, "target": TARGETS, "heldout": HELDOUT}[cohort]
    ranks = read(CACHED_RANKS if cohort == "public" else output / f"{cohort}-rankings.json")
    path = output / f"{cohort}-{'retrieval' if retrieval_only else 'qa'}.json"
    results = read(path) if path.exists() else []
    documents = read(manifest)["documents"]
    questions = {(doc["id"], q["id"]): q for doc in documents for q in doc["questions"]}
    for row in results:
        if "answer" in row:
            row.update(score_answer(questions[(row["document"], row["id"])], row["answer"]))
    completed = {(r["document"], r["id"], r["method"]) for r in results}
    reader = None
    if not retrieval_only:
        from eval.granularity_reader import Reader

        reader = Reader(SIZE)
    for raw in documents:
        layout, layout_path = page_layout(raw)
        prepared = {}
        with fitz.open(ROOT / raw["pdf"]) as pdf:
            for q in raw["questions"]:
                question = q["question"]
                cached = ranks[raw["id"]]["questions"][question]["evidence"]
                pages = tuple(item["page"] for item in cached)
                if pages not in prepared:
                    prepared[pages] = make_views(pdf, pages, layout)
                views = prepared[pages]
                pending = [
                    method for method in methods if (raw["id"], q["id"], method) not in completed
                ]
                verification = None
                if (
                    reader is not None
                    and q.get("answers")
                    and any(
                        method in {"verification-control", "verification"} for method in pending
                    )
                ):
                    verification = verification_views(pdf, question, views, reader)
                for method in pending:
                    key = (raw["id"], q["id"], method)
                    started = perf_counter()
                    uses_verification = method in {"verification-control", "verification"}
                    if uses_verification and verification is not None:
                        selected = verification[int(method == "verification")]
                    else:
                        selection_method = "heading-scope" if uses_verification else method
                        selected = choose(question, views, selection_method)
                    elapsed = perf_counter() - started
                    row = {
                        "document": raw["id"],
                        "id": q["id"],
                        "method": method,
                        "question": question,
                        "candidate_pages": [p + 1 for p in pages],
                        "candidate_covers_all_pages": set(q.get("evidence_pages_raw", []))
                        <= {p + 1 for p in pages},
                        "selected": [
                            {"id": v.id, "page": v.page + 1, "bbox": v.bbox, "kind": v.kind}
                            for v in selected
                        ],
                        "layout": layout_path,
                        "retrieval_seconds": elapsed,
                        "gold_pages": q.get("evidence_pages_raw", []),
                        "all_pages": set(q.get("evidence_pages_raw", []))
                        <= {v.page + 1 for v in selected},
                    }
                    if "target_bbox" in q:
                        row.update(
                            target_visible=coverage(
                                selected, q["evidence_pages_raw"][0], q["target_bbox"]
                            ),
                            distractor_visible=any(
                                coverage(selected, p, b) for p, b in q["distractors"]
                            ),
                            has_distractor=bool(q["distractors"]),
                            pair=q.get("pair"),
                        )
                    if reader is not None and q.get("answers"):
                        if method in {"text-layout", "heading-text"}:
                            answer = reader.answer(question, [], text_evidence(pdf, selected))
                            budget = 0
                        else:
                            images = [render(pdf, v) for v in selected]
                            budget = 6 if method == "colpali-top6" else 2
                            images += [Image.new("RGB", SIZE, "white")] * (budget - len(images))
                        answer = reader.answer(question, images)
                        if answer["visual_tokens"] != VISUAL_TOKENS * budget:
                            raise RuntimeError("Qwen processed a different visual token budget")
                        if uses_verification and verification is not None:
                            observation = verification[2]
                            row["verification"] = observation
                            for metric in ("visual_tokens", "prompt_tokens", "generation_seconds"):
                                answer[metric] += observation[metric]
                        row.update(answer, **score_answer(q, answer["answer"]))
                        if "answer_format" in q:
                            from eval.official import mmlongbench

                            row["official_score"] = mmlongbench(q, answer["answer"])
                    results.append(row)
                    save(path, results)
                    completed.add(key)
            print(cohort, raw["id"], len(results), flush=True)
    save(path, results)
    summary = {}
    for method in ALL_METHODS:
        rows = [row for row in results if row["method"] == method]
        if not rows:
            continue
        evidence = [row for row in rows if row["gold_pages"]]
        comparisons = [row for row in rows if row.get("has_distractor")]
        pairs = defaultdict(list)
        for row in rows:
            if row.get("pair"):
                pairs[(row["document"], row["pair"])].append(row)
        summary[method] = {
            "n": len(rows),
            "evidence_n": len(evidence),
            "all_pages": sum(row["all_pages"] for row in evidence),
            "target_visible": sum(row.get("target_visible", False) for row in rows),
            "with_distractor_n": len(comparisons),
            "target_only": sum(
                row["target_visible"] and not row["distractor_visible"] for row in comparisons
            ),
            "pairs_answer_correct": sum(
                len(pair) == 2 and all(row.get("accuracy") == 1 for row in pair)
                for pair in pairs.values()
            ),
            "pairs_region_and_answer_correct": sum(
                len(pair) == 2
                and all(
                    row.get("accuracy") == 1
                    and row.get("target_visible")
                    and not row.get("distractor_visible")
                    for row in pair
                )
                for pair in pairs.values()
            ),
            "pairs_n": len(pairs),
        }
        for metric in (
            "accuracy",
            "anls",
            "official_score",
            "visual_tokens",
            "prompt_tokens",
            "generation_seconds",
        ):
            values = [row[metric] for row in rows if metric in row]
            if values:
                summary[method][metric] = mean(values)
                summary[method]["scored"] = len(values)
    save(path.with_name(path.stem + "-summary.json"), summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("index", "retrieval", "qa"))
    parser.add_argument("--cohort", choices=("public", "target", "heldout"), default="target")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", choices=ALL_METHODS, nargs="+", default=METHODS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "index":
        index(args.output, args.cohort)
    else:
        experiment(args.output, args.cohort, args.stage == "retrieval", args.methods)


if __name__ == "__main__":
    main()
