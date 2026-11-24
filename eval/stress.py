import random
from pathlib import Path

from PIL import Image

from src.document import Document, MacroNode, Mention
from src.layout import native_region

LEVELS = (0, 1, 2, 4)


def select_documents(documents: list[dict], size: int, seed: int) -> list[dict]:
    eligible = sorted(
        (doc for doc in documents if len(doc.get("distractors", [])) >= max(LEVELS)),
        key=lambda doc: doc["id"],
    )
    if len(eligible) < size:
        raise ValueError(f"Stress test requires {size} eligible documents. Found {len(eligible)}.")
    rng = random.Random(seed)
    return [
        {**doc, "distractors": rng.sample(doc["distractors"], k=max(LEVELS))}
        for doc in rng.sample(eligible, k=size)
    ]


def inject(
    document: Document,
    labels: dict[str, str],
    specs: list[dict],
    root: Path,
    level: int,
) -> tuple[Document, dict[str, str]]:
    pages, macros, mentions, labels = (
        dict(document.pages),
        list(document.macro),
        list(document.mentions),
        dict(labels),
    )
    originals = {node.id: node for node in document.mentions if node.kind == "entity"}
    used_paths = set()
    for i, spec in enumerate(specs[:level]):
        target = originals.get(spec["target"])
        if target is None or target.region is None:
            raise ValueError(
                f"{document.id}: distractor target {spec['target']} needs a grounded entity"
            )
        page = max(pages) + 1
        asset = root / spec["image"]
        if asset.resolve() in used_paths or asset.resolve() in {
            path.resolve() for path in document.pages.values()
        }:
            raise ValueError(f"{document.id}: distractor pages must use distinct assets")
        used_paths.add(asset.resolve())
        with Image.open(asset) as image:
            region = native_region(page, spec["bbox"], image.size, "pixels")
        key = f"__distractor_{i + 1}"
        pages[page] = asset
        macros.append(
            MacroNode(key, "figure", spec["caption"], page, region, spec.get("section", ""))
        )
        mentions.append(Mention(f"{key}:mention", key, target.text, page, region))
        gold_id = spec["gold_entity"]
        if target.id not in labels or gold_id == labels[target.id]:
            raise ValueError(
                f"{document.id}: label the target and provide a different distractor identity"
            )
        labels[f"{key}:mention"] = gold_id
    return Document(document.id, pages, tuple(macros), tuple(mentions), document.source_pdf), labels
