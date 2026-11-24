"""Build a matched polysemy manifest from alternate figures in the benchmark corpus."""

import argparse
import hashlib
import json
import os
import random
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.document import Document, Mention, Region
from src.layout import load_document


def _fingerprint(image: Image.Image) -> np.ndarray:
    return np.asarray(image.resize((24, 24)), dtype=np.float32)


def _page(image: Image.Image, name: str, destination: Path) -> list[int]:
    page = Image.new("RGB", (1000, 1250), "white")
    drawing = ImageDraw.Draw(page)
    font = ImageFont.load_default(size=28)
    caption = "\n".join(textwrap.wrap(name, width=50))
    drawing.multiline_text((65, 70), caption, fill="black", font=font, spacing=5)
    crop = ImageOps.contain(image, (850, 930))
    left, top = (1000 - crop.width) // 2, 210
    page.paste(crop, (left, top))
    page.save(destination)
    return [left, top, left + crop.width, top + crop.height]


def _target(
    document: Document, labels: dict[str, str], evidence_pages: set[int] | None = None
) -> Mention | None:
    entities = [
        node
        for node in document.mentions
        if node.kind == "entity"
        and node.region is not None
        and 2 <= len(node.text.strip()) <= 120
        and (evidence_pages is None or node.region.page + 1 in evidence_pages)
    ]
    return min(entities, key=lambda node: (node.id not in labels, node.page, node.id), default=None)


def build_pool(
    manifest: Path,
    output: Path,
    cache: Path,
    size: int,
    seed: int,
    dpi: int,
    min_visual_difference: float,
    *,
    evidence_questions_only: bool = False,
) -> int:
    source = json.loads(manifest.read_text())
    candidates = []
    donors: dict[str, tuple[Document, Region, np.ndarray]] = {}
    for raw in sorted(source["documents"], key=lambda item: item["id"]):
        doc = load_document(raw, manifest.parent, cache, dpi)
        macro = next((node for node in doc.macro if node.kind == "figure" and node.region), None)
        if macro is not None:
            with doc.crop(macro.region) as image:
                donors[doc.id] = (doc, macro.region, _fingerprint(image))
        evidence_pages = (
            {
                page
                for question in raw["questions"]
                for page in question.get("evidence_pages_raw", [])
            }
            if evidence_questions_only
            else None
        )
        mention = _target(doc, raw.get("entity_gold", {}), evidence_pages)
        if mention is not None:
            candidates.append((raw, doc, mention))
    print(
        f"Grounded candidate documents: {len(candidates)}. Visual donor documents: {len(donors)}."
    )
    rng = random.Random(seed)
    rng.shuffle(candidates)
    chosen = []
    for raw, doc, mention in candidates:
        with doc.crop(mention.region) as image:
            reference = _fingerprint(image)
        options = [
            key
            for key, (_, _, fingerprint) in donors.items()
            if key != doc.id and np.mean(np.abs(reference - fingerprint)) >= min_visual_difference
        ]
        if len(options) >= 4:
            chosen.append((raw, mention, rng.sample(options, k=4)))
        if len(chosen) == size:
            break
    if len(chosen) < size:
        raise ValueError(f"Found {len(chosen)} eligible targets with four donors, required {size}")

    output.parent.mkdir(parents=True, exist_ok=True)
    assets = output.parent / (output.stem + "-pages")
    assets.mkdir(exist_ok=True)

    def relative(value: str) -> str:
        return os.path.relpath((manifest.parent / value).resolve(), output.parent.resolve())

    documents = []
    for raw, target, alternatives in chosen:
        modified = {**raw, "entity_gold": dict(raw.get("entity_gold", {})), "distractors": []}
        modified["entity_gold"].setdefault(target.id, f"original:{target.id}")
        for key in ("pdf", "layout"):
            if key in modified:
                modified[key] = relative(modified[key])
        if "pages" in modified:
            pages = modified["pages"]
            modified["pages"] = (
                {key: relative(value) for key, value in pages.items()}
                if isinstance(pages, dict)
                else [relative(value) for value in pages]
            )
        questions = raw["questions"]
        if evidence_questions_only:
            questions = [
                question
                for question in questions
                if target.region.page + 1 in question.get("evidence_pages_raw", [])
            ]
        modified["questions"] = [
            {**question, "image": relative(question["image"])}
            if question.get("image")
            else question
            for question in questions
        ]
        prefix = hashlib.sha256(raw["id"].encode()).hexdigest()[:20]
        for i, donor_id in enumerate(alternatives, 1):
            donor, region, _ = donors[donor_id]
            path = assets / f"{prefix}-{i}.png"
            with donor.crop(region) as image:
                bbox = _page(image, target.text, path)
            modified["distractors"].append(
                {
                    "target": target.id,
                    "image": str(path.relative_to(output.parent)),
                    "bbox": bbox,
                    "caption": target.text,
                    "gold_entity": f"distractor:{prefix}:{i}",
                    "donor_document": donor_id,
                    "donor_page": region.page,
                    "donor_bbox": region.bbox,
                }
            )
        documents.append(modified)
    output.write_text(
        json.dumps(
            {
                **source,
                "documents": documents,
                "stress_preparation": {
                    "source": str(manifest.resolve()),
                    "size": size,
                    "seed": seed,
                    "min_visual_difference": min_visual_difference,
                    "dpi": dpi,
                    "evidence_questions_only": evidence_questions_only,
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    return len(documents)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=250)
    parser.add_argument("--min-visual-difference", type=float, default=20)
    parser.add_argument("--render-cache", type=Path, default=Path("data/rendered"))
    parser.add_argument(
        "--evidence-questions-only",
        action="store_true",
        help="Choose grounded targets on evidence pages and retain questions citing the target page",
    )
    options = parser.parse_args()
    if options.size < 1 or options.dpi < 1 or not 0 < options.min_visual_difference <= 255:
        parser.error("Expected positive size and DPI, and visual difference in (0,255]")
    build_pool(
        options.manifest,
        options.output,
        options.render_cache,
        options.size,
        options.seed,
        options.dpi,
        options.min_visual_difference,
        evidence_questions_only=options.evidence_questions_only,
    )


if __name__ == "__main__":
    main()
