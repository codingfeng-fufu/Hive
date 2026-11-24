"""Replay document entity pairs in six fixed Qwen conditions, then score the batch."""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from contextlib import ExitStack
from fractions import Fraction
from pathlib import Path
from time import perf_counter
from urllib.request import Request, urlopen

import fitz
from PIL import ImageDraw
from scope_pair_probe import distance, page_lines, rect

from eval.vger_pilot import STRATA, _crop
from src.generation import _image

VIEWS = ("text", "figure", "crop")
METHODS = tuple(f"{view}{suffix}" for suffix in ("", "_references") for view in VIEWS)
REFERENCE = re.compile(r"\bfig(?:ure)?\.?\s+(\d+(?:[-.]\d+)?[a-z]?)\b", re.IGNORECASE)
PROMPT = """Resolve the identity of the two marked document mentions. The same object,
part, or data series can occur under different names. Different parts of the same assembly
and different series using the same metric are distinct. Drawings that look alike or labels
that repeat do not establish identity by themselves.
For each endpoint use its own context. An image, if supplied, shows that endpoint's figure
or local region. Use an explicit identifier, reference, part or series scope when available.
Return one JSON object with relation same, different, or unclear, and a concise reason.
Include the source IDs you relied on in a sources array. Only cite supplied IDs."""


def read_cases(legacy, expansion, seed):
    legacy_rows = json.loads(legacy.read_text())["rows"]
    expansion_rows = json.loads(expansion.read_text())["rows"]
    seed_rows = json.loads(seed.read_text())["rows"]
    inputs = (
        ("reanalysis", [row for row in legacy_rows if row["split"] == "heldout"]),
        ("mechanism_seed", [row for row in seed_rows if row["split"] == "reference_seed"]),
        ("new_manual", [row for row in expansion_rows if row["split"] == "reference_seed"]),
    )
    for cohort, rows in inputs:
        for row in rows:
            identity = [
                row["document"],
                *(
                    (row[f"{side}_page"], row[f"{side}_bbox"], row[f"{side}_label"])
                    for side in ("left", "right")
                ),
            ]
            key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:16]
            yield dict(row, id=key, cohort=cohort)


def source_lines(pdf, page_number, figure_box, local_text, cached_pages):
    primary = pdf[page_number - 1]
    parent = rect(primary, figure_box)

    def lines(n):
        if n not in cached_pages:
            cached_pages[n] = page_lines(pdf[n - 1])
        return cached_pages[n]

    nearby = [(box, text) for box, text in lines(page_number) if distance(box, parent)[0] <= 36]
    nearby.sort(key=lambda entry: distance(entry[0], parent))
    referenced = {number.casefold() for number in REFERENCE.findall(local_text)}
    cited = []
    caption = []
    for _, text in nearby:
        numbers = {number.casefold() for number in REFERENCE.findall(text)}
        if numbers:
            caption.append((page_number, text))
            referenced.update(numbers)
    if referenced:
        for n in range(max(1, page_number - 2), min(len(pdf), page_number + 2) + 1):
            cited.extend(
                (n, text)
                for _, text in lines(n)
                if referenced & {number.casefold() for number in REFERENCE.findall(text)}
            )
    sources = [*caption, *cited, *((page_number, text) for _, text in nearby[:4])]
    return list(dict.fromkeys(sources))[:8]


def make_request(row, method, pdf, cached_pages):
    view = method.removesuffix("_references")
    messages = []
    evidence_ids = set()
    for side in ("left", "right"):
        page = row[f"{side}_page"]
        source = f"{side}-local"
        evidence_ids.add(source)
        local = row[f"{side}_source_context"]
        text = f"{side.upper()} | mention: {row[f'{side}_label']} | page {page}\n"
        text += f"[{source}] {local}"
        if method.endswith("_references"):
            context = source_lines(pdf, page, row[f"{side}_figure_bbox"], local, cached_pages)
            for i, (source_page, content) in enumerate(context, start=1):
                source = f"{side}-reference-{i}"
                evidence_ids.add(source)
                text += f"\n[{source}] Page {source_page}: {content}"
        messages.append({"type": "text", "text": text})
        if view != "text":
            key = "figure_bbox" if view == "figure" else "bbox"
            box = row[f"{side}_{key}"]
            image = _crop(pdf, page, box)
            if view == "figure":
                x0, y0, x1, y1 = box
                left, top, right, bottom = row[f"{side}_bbox"]
                width, height = image.size
                outline = (
                    max(0, round((left - x0) * width / (x1 - x0))),
                    max(0, round((top - y0) * height / (y1 - y0))),
                    min(width - 1, round((right - x0) * width / (x1 - x0))),
                    min(height - 1, round((bottom - y0) * height / (y1 - y0))),
                )
                ImageDraw.Draw(image).rectangle(outline, outline="#db1853", width=3)
            messages.append(_image(image, None, ""))
    return [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": messages},
    ], evidence_ids


def infer(url, model, messages, evidence_ids):
    body = {
        "model": model,
        "messages": messages,
        "seed": 42,
        "temperature": 0,
        "max_tokens": 128,
    }
    request = Request(
        url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = perf_counter()
    with urlopen(request, timeout=90) as response:
        output = json.load(response)
    choice = output["choices"][0]
    raw = choice["message"]["content"] or ""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        parsed = json.loads(match[0]) if match else {}
    except json.JSONDecodeError:
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}
    relation = parsed.get("relation")
    if relation not in ("same", "different", "unclear"):
        relation = "invalid"
    sources = parsed.get("sources")
    cited = (
        [source for source in sources if isinstance(source, str)]
        if isinstance(sources, list)
        else []
    )
    return {
        "raw": raw,
        "relation": relation,
        "reason": parsed.get("reason"),
        "sources": cited,
        "citations_valid": bool(cited) and set(cited) <= evidence_ids,
        "usage": output.get("usage"),
        "finish_reason": choice.get("finish_reason"),
        "latency_seconds": round(perf_counter() - started, 3),
    }


def summarize(rows, predictions):
    summary = {}
    for cohort in dict.fromkeys(row["cohort"] for row in rows):
        subset = [row for row in rows if row["cohort"] == cohort]
        summary[cohort] = {}
        for method in METHODS:
            counts = Counter()
            for row in subset:
                result = predictions[row["id"], method]
                counts[row["stratum"]] += 1
                counts[f"{row['stratum']}_links"] += result["relation"] == "same"
                counts["invalid"] += result["relation"] == "invalid"
                counts["unclear"] += result["relation"] == "unclear"
                counts["citations_valid"] += result["citations_valid"]
                counts["prompt_tokens"] += (result["usage"] or {}).get("prompt_tokens") or 0
            links = counts[f"{STRATA[2]}_links"] + counts[f"{STRATA[3]}_links"]
            summary[cohort][method] = {
                "false_merges": [counts[f"{STRATA[0]}_links"], counts[STRATA[0]]],
                "same_name_retained": [counts[f"{STRATA[1]}_links"], counts[STRATA[1]]],
                "alias_precision": [counts[f"{STRATA[2]}_links"], links],
                "alias_recall": [counts[f"{STRATA[2]}_links"], counts[STRATA[2]]],
                "invalid": counts["invalid"],
                "unclear": counts["unclear"],
                "citations_with_valid_ids": counts["citations_valid"],
                "prompt_tokens": counts["prompt_tokens"],
            }
    return summary


def paired_changes(rows, predictions):
    comparisons = {}
    for baseline, method in (
        ("text", "figure"),
        ("text", "crop"),
        ("figure", "crop"),
        ("text", "text_references"),
        ("figure", "figure_references"),
        ("crop", "crop_references"),
        ("text_references", "figure_references"),
        ("text_references", "crop_references"),
        ("figure_references", "crop_references"),
    ):
        documents = defaultdict(Counter)
        for row in rows:
            before = predictions[row["id"], baseline]["relation"]
            after = predictions[row["id"], method]["relation"]
            expected = "same" if row["stratum"] in (STRATA[1], STRATA[2]) else "different"
            key = "unchanged"
            if before != expected and after == expected:
                key = "fixed"
            elif before == expected and after != expected:
                key = "regressed"
            documents[row["document"]][f"{row['stratum']}:{key}"] += 1
        comparisons[f"{baseline}->{method}"] = documents
    return comparisons


def decide(rows, predictions, counts, changes):
    cohort = counts["reanalysis"]
    same_negative = STRATA[0]
    alias_strata = STRATA[2:]
    routes = {}
    for view in ("figure", "crop"):
        for suffix in ("", "_references"):
            method, baseline = f"{view}{suffix}", f"text{suffix}"
            before, after = cohort[baseline], cohort[method]
            by_doc = changes[f"{baseline}->{method}"]
            doc_ids = {row["document"] for row in rows if row["cohort"] == "reanalysis"}
            split_docs = sorted(
                doc
                for doc in doc_ids
                if by_doc[doc][f"{same_negative}:fixed"] > by_doc[doc][f"{same_negative}:regressed"]
            )
            alias_docs = sorted(
                doc
                for doc in doc_ids
                if sum(
                    by_doc[doc][f"{s}:fixed"] - by_doc[doc][f"{s}:regressed"] for s in alias_strata
                )
                > 0
            )
            measured = [row for row in rows if row["cohort"] == "reanalysis"]
            abstentions = {
                variant: sum(
                    predictions[row["id"], variant]["relation"] in ("invalid", "unclear")
                    for row in measured
                )
                for variant in (baseline, method)
            }
            split_improved = (
                after["false_merges"][0] < before["false_merges"][0]
                and after["same_name_retained"][0] >= before["same_name_retained"][0]
                and len(split_docs) >= 2
            )
            after_tp, after_links = after["alias_precision"]
            before_tp, before_links = before["alias_precision"]
            alias_improved = (
                after_tp >= before_tp
                and after_links > 0
                and (
                    not before_links
                    or Fraction(after_tp, after_links) >= Fraction(before_tp, before_links)
                )
                and (after_tp > before_tp or after_links < before_links)
                and len(alias_docs) >= 2
            )
            routes[method] = {
                "baseline": baseline,
                "split_fixed_net_documents": split_docs,
                "alias_fixed_net_documents": alias_docs,
                "abstentions": abstentions,
                "split_gate": split_improved and abstentions[method] <= abstentions[baseline],
                "alias_gate": alias_improved and abstentions[method] <= abstentions[baseline],
            }
    keep = [name for name, route in routes.items() if route["split_gate"] or route["alias_gate"]]
    return {
        "eligible_for_independent_replication": keep,
        "next_step": (
            "Verify source traces, then replicate the passing component on new PDFs."
            if keep
            else "Keep the text baseline and inspect misses before expanding the visual mechanism."
        ),
        "visual_routes": routes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--expansion", type=Path, required=True)
    parser.add_argument("--pdfs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    rows = list(read_cases(args.legacy, args.expansion, args.seed))
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Expected a nonempty batch of unique pairs")
    signature = hashlib.sha256()
    for path in (
        args.legacy,
        args.seed,
        args.expansion,
        Path(__file__).resolve(),
        Path(__file__).with_name("scope_pair_probe.py"),
    ):
        signature.update(path.read_bytes())
    signature.update(args.model.encode())
    for name in sorted({row["document"] for row in rows}):
        with (args.pdfs / f"{name}.pdf").open("rb") as pdf:
            for block in iter(lambda: pdf.read(1024 * 1024), b""):
                signature.update(block)
    fingerprint = signature.hexdigest()

    log = args.out / "predictions.jsonl"
    predictions = {}
    if log.exists():
        # An interrupted write may leave a partial last line. Keep all complete records.
        with log.open("rb+") as previous:
            while line := previous.readline():
                if not line.endswith(b"\n"):
                    previous.truncate(previous.tell() - len(line))
                    break
                item = json.loads(line)
                if item["fingerprint"] != fingerprint:
                    raise ValueError("Existing predictions came from different frozen inputs")
                predictions[item["id"], item["method"]] = item
    print(f"{len(rows)} pairs, {len(METHODS)} routes, {len(predictions)} calls cached", flush=True)
    with ExitStack() as stack, log.open("a", encoding="utf-8") as stream:
        pdfs = {}
        page_caches = defaultdict(dict)
        for row in rows:
            name = row["document"]
            if name not in pdfs:
                pdfs[name] = stack.enter_context(fitz.open(args.pdfs / f"{name}.pdf"))
            for method in METHODS:
                key = row["id"], method
                if key in predictions:
                    continue
                messages, ids = make_request(row, method, pdfs[name], page_caches[name])
                result = infer(args.url, args.model, messages, ids)
                result["input_text"] = [
                    part["text"] for part in messages[1]["content"] if part["type"] == "text"
                ]
                result.update(id=row["id"], method=method, fingerprint=fingerprint)
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
                predictions[key] = result
            print(row["cohort"], name, row["id"], flush=True)
    counts = summarize(rows, predictions)
    changes = paired_changes(rows, predictions)
    report = {
        "fingerprint": fingerprint,
        "model": args.model,
        "pairs": len(rows),
        "calls": len(predictions),
        "cases": rows,
        "counts": counts,
        "paired_changes_by_document": changes,
        "decision": decide(rows, predictions, counts, changes),
    }
    temporary = args.out / "summary.tmp"
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(args.out / "summary.json")
    print(json.dumps(report["counts"], indent=2), flush=True)


if __name__ == "__main__":
    main()
