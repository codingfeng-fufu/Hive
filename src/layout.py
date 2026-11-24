import hashlib
import json
import math
import re
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

import fitz
from PIL import Image

from .document import Document, MacroNode, Mention, Region

_LABEL = re.compile(r"\b(fig(?:ure)?|table)\.?\s*(\d+(?:[.-]\d+)*[a-z]?)\b", re.IGNORECASE)
_NAME = re.compile(
    r"\b(?:[A-Z][A-Za-z-]+(?:\s+[A-Z][A-Za-z-]+){1,4}|"
    r"(?:[A-Za-z][A-Za-z-]*\s+){1,3}#?(?:[A-Z]|[A-Z]*\d{2,}[A-Z\d-]*))\b"
)


_COMPONENT = re.compile(
    r"\b(?:the|a|an) ((?:[a-z][a-z-]* ){1,3}"
    r"(?:assembly|unit|module|valve|controller|pump|sensor|circuit|connector|motor))\b",
    re.IGNORECASE,
)
_PART = re.compile(r"\b(?:part|item)\s*#\s*[A-Z0-9][A-Z0-9-]*", re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<!Fig\.)(?<!fig\.)(?<=[.!?])\s+")


def _names(text: str) -> list[str]:
    values = [*(_PART.findall(text)), *_NAME.findall(text), *_COMPONENT.findall(text)]
    return list(dict.fromkeys(name for name in values if not _LABEL.fullmatch(name)))


def _reference_target(
    label: tuple[str, str],
    parent: MacroNode,
    page: int,
    captions: dict[tuple[str, str], list[MacroNode]],
) -> MacroNode | None:
    candidates = [node for node in captions.get(label, []) if node.id != parent.id]
    if not candidates:
        return None

    def distance(node: MacroNode) -> tuple[bool, int]:
        return node.section != parent.section, abs(node.page - page)

    closest = min(candidates, key=distance)
    return closest if sum(distance(node) == distance(closest) for node in candidates) == 1 else None


def native_region(page: int, bbox: list, size: tuple[int, int], scale: str) -> Region:
    width, height = size
    factors = (width / 1000, height / 1000) if scale == "normalized_1000" else (1, 1)
    x0, y0, x1, y1 = [value * factors[index % 2] for index, value in enumerate(bbox)]
    box = (
        max(0, math.floor(x0)),
        max(0, math.floor(y0)),
        min(width, math.ceil(x1)),
        min(height, math.ceil(y1)),
    )
    if not (box[0] < box[2] and box[1] < box[3]):
        raise ValueError(f"Empty bbox on page {page}: {bbox}")
    return Region(page, box)


def _labels(text: str) -> list[tuple[str, str]]:
    return [
        ("table" if kind.lower() == "table" else "figure", number.lower())
        for kind, number in _LABEL.findall(text)
    ]


def _render(pdf: Path, cache: Path, dpi: int) -> dict[int, Path]:
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()[:20]
    target = cache / f"{digest}-{dpi}"
    target.mkdir(parents=True, exist_ok=True)
    with fitz.open(pdf) as source:
        paths = {i: target / f"{i:05}.png" for i in range(len(source))}
        for index, path in paths.items():
            if not path.exists():
                with tempfile.TemporaryDirectory(dir=target) as staging:
                    temporary = Path(staging) / path.name
                    source[index].get_pixmap(dpi=dpi, alpha=False).save(temporary)
                    temporary.replace(path)
    return paths


def parse_pdf(pdf: Path, output: Path, backend: str = "pipeline") -> Path:
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(["mineru", "-p", str(pdf), "-o", str(output), "-b", backend], check=True)
    results = sorted((output / pdf.stem).rglob(f"{pdf.stem}_content_list.json"))
    if len(results) != 1:
        raise ValueError(f"Expected one MinerU content_list for {pdf}, found {len(results)}")
    return results[0]


def _nearby_caption(
    region: Region,
    blocks: list[dict],
    size: tuple[int, int],
    scale: str,
) -> str:
    x0, y0, x1, _ = region.bbox
    width, height = size
    candidates = []
    for block in blocks:
        text = str(block.get("text") or "").strip()
        if block["type"] != "text" or not block.get("bbox") or not 0 < len(text) <= 180:
            continue
        try:
            left, _, right, bottom = native_region(region.page, block["bbox"], size, scale).bbox
        except ValueError:
            continue
        gap = y0 - bottom
        overlap = min(x1, right) - max(x0, left)
        if (
            0 <= gap <= 0.12 * height
            and abs(left - x0) <= 0.08 * width
            and overlap >= 0.65 * min(x1 - x0, right - left)
        ):
            candidates.append((gap, text))
    return " ".join(text for _, text in reversed(sorted(candidates)[:2]))


def _subregions(
    page: fitz.Page | None,
    region: Region,
    size: tuple[int, int],
    kind: str,
    caption: str,
) -> list[tuple[Region, str]]:
    if page is not None and page.rotation == 0:
        width, height = size
        sx, sy = page.rect.width / width, page.rect.height / height
        x0, y0, x1, y1 = region.bbox
        clip = fitz.Rect(x0 * sx, y0 * sy, x1 * sx, y1 * sy)

        def convert(box: fitz.Rect) -> Region:
            cropped = fitz.Rect(box) & clip
            return native_region(
                region.page,
                [
                    cropped.x0 / sx,
                    cropped.y0 / sy,
                    cropped.x1 / sx,
                    cropped.y1 / sy,
                ],
                size,
                "pixels",
            )

        if kind == "table":
            blocks = []
            for table in page.find_tables(clip=clip).tables:
                for index, (row, values) in enumerate(zip(table.rows, table.extract()), 1):
                    text = " | ".join(value for value in values if value)
                    blocks.append((convert(row.bbox), f"Table row {index}. {text} {caption}"))
            if blocks:
                return blocks
        else:
            blocks = []
            for image in page.get_image_info():
                box = fitz.Rect(image["bbox"])
                overlap = (box & clip).get_area()
                if 0.08 * clip.get_area() <= overlap <= 0.8 * clip.get_area():
                    blocks.append((convert(box), caption))
            if len(blocks) >= 2:
                return list(dict.fromkeys(blocks))
    x0, y0, x1, y1 = region.bbox
    w, h = x1 - x0, y1 - y0
    if min(w, h) < 32:
        return [(region, caption)]
    blocks = []
    for vertical, v in enumerate((0, 0.40)):
        for horizontal, u in enumerate((0, 0.40)):
            descriptor = (
                ("top" if vertical == 0 else "bottom")
                + "-"
                + ("left" if horizontal == 0 else "right")
            )
            box = (
                x0 + int(w * u),
                y0 + int(h * v),
                x0 + math.ceil(w * (u + 0.60)),
                y0 + math.ceil(h * (v + 0.60)),
            )
            blocks.append((Region(region.page, box), f"{descriptor} {kind} region. {caption}"))
    return blocks


def _summarize(paragraphs: list[str]) -> str:
    sentences = [
        sentence.strip()
        for text in paragraphs
        for sentence in _SENTENCE_END.split(text)
        if sentence.strip()
    ]
    return " ".join(sentences[:2])[:750]


def read_mineru(
    doc_id: str,
    pages: dict[int, Path],
    layout: Path,
    pdf: Path | None = None,
) -> Document:
    payload = json.loads(layout.read_text())
    blocks = payload["blocks"] if isinstance(payload, dict) else payload
    scale = (
        payload.get("coordinate_space", "normalized_1000")
        if isinstance(payload, dict)
        else "normalized_1000"
    )
    if scale not in {"normalized_1000", "pixels"}:
        raise ValueError(f"Unsupported MinerU bbox coordinate space: {scale}")
    sizes = {}
    for index, path in pages.items():
        with Image.open(path) as image:
            sizes[index] = image.size
    macros, mentions = [], []
    page_blocks: dict[int, list[dict]] = {}
    for block in blocks:
        page_blocks.setdefault(int(block["page_idx"]), []).append(block)
    section_text: dict[str, list[str]] = {}
    caption_labels: dict[tuple[str, str], list[MacroNode]] = {}
    pending_grounding: list[tuple[int, MacroNode, tuple[str, str]]] = []
    source = fitz.open(pdf) if pdf else None
    chapter = ""
    section = None
    try:
        for i, block in sorted(enumerate(blocks), key=lambda item: (item[1]["page_idx"], item[0])):
            page = int(block["page_idx"])
            kind = block["type"]
            if kind not in {"text", "list", "image", "table", "chart"}:
                continue
            key = f"p{page:04}:b{i:05}"
            text = str(block.get("text", "")).strip()
            if kind in {"text", "list"} and int(block.get("text_level", 0) or 0) > 0:
                chapter = key
                section = None
            if section is None:
                section = MacroNode(f"{key}:section", "section", "", page, section=chapter)
                section_text[section.id] = []
                macros.append(section)
            if kind in {"text", "list"}:
                if not text:
                    text = " ".join(map(str, block.get("list_items", []))).strip()
                if not text:
                    continue
                section_text[section.id].append(text)
                entities = {}
                for j, sentence in enumerate(_SENTENCE_END.split(text)):
                    if not sentence.strip():
                        continue
                    mentions.append(
                        Mention(
                            f"{key}:t{j}",
                            section.id,
                            sentence.strip(),
                            page,
                            kind="span",
                        )
                    )
                    refs = _labels(sentence)
                    names = _names(sentence)
                    for name in names:
                        if name in entities:
                            continue
                        mention_id = f"{key}:e{len(entities)}"
                        entities[name] = mention_id
                        if len(refs) == len(names) == 1:
                            pending_grounding.append((len(mentions), section, refs[0]))
                        mentions.append(Mention(mention_id, section.id, name, page))
                continue
            kind = "table" if kind == "table" else "figure"
            prefix = "table" if kind == "table" else "image"
            captions = block.get(f"{prefix}_caption") or block.get("caption") or []
            caption = " ".join(captions) if isinstance(captions, list) else str(captions)
            region = native_region(page, block["bbox"], sizes[page], scale)
            caption = (
                caption
                or text
                or _nearby_caption(region, page_blocks[page], sizes[page], scale)
                or kind.capitalize()
            )
            macro = MacroNode(key, kind, caption, page, region, chapter)
            macros.append(macro)
            labels = _labels(caption)
            if labels:
                caption_labels.setdefault(labels[0], []).append(macro)
            for j, name in enumerate(_names(caption)):
                mentions.append(Mention(f"{key}:e{j}", key, name, page, region))
            explicit = block.get("regions")
            if explicit:
                for j, child in enumerate(explicit):
                    child_box = native_region(page, child["bbox"], sizes[page], scale)
                    child_text = child.get("text", caption)
                    child_kind = "entity" if child.get("entity", False) else "region"
                    mentions.append(
                        Mention(f"{key}:r{j}", key, child_text, page, child_box, child_kind)
                    )
            else:
                visual_page = source[page] if source else None
                for j, (box, label) in enumerate(
                    _subregions(visual_page, region, sizes[page], kind, caption)
                ):
                    mentions.append(Mention(f"{key}:r{j}", key, label, page, box, "region"))
    finally:
        if source is not None:
            source.close()
    if not macros:
        raise ValueError(f"{doc_id}: layout parsing produced no supported blocks")
    for index, section, label in pending_grounding:
        occurrence = mentions[index]
        target = _reference_target(label, section, occurrence.page, caption_labels)
        if target is not None:
            mentions[index] = replace(occurrence, parent=target.id, region=target.region)
    output = []
    for macro in macros:
        paragraphs = section_text.get(macro.id, [macro.text])
        targets = set()
        refs = _labels(" ".join(paragraphs))
        if macro.region and refs and macro in caption_labels.get(refs[0], []):
            refs = refs[1:]
        for label in refs:
            target = _reference_target(label, macro, macro.page, caption_labels)
            if target is not None:
                targets.add(target.id)
        if macro.kind == "section":
            macro = replace(
                macro, text=_summarize(paragraphs) or f"Section on page {macro.page + 1}"
            )
        output.append(replace(macro, references=tuple(sorted(targets))))
    return Document(doc_id, pages, tuple(output), tuple(mentions), pdf)


def load_document(
    raw: dict,
    root: Path,
    cache: Path,
    dpi: int,
    *,
    require_graph: bool = True,
) -> Document:
    pdf = root / raw["pdf"] if "pdf" in raw else None
    if "pages" in raw:
        paths = raw["pages"]
        pages = {
            int(i): root / path
            for i, path in (paths.items() if isinstance(paths, dict) else enumerate(paths))
        }
    elif pdf:
        pages = _render(pdf, cache, dpi)
    else:
        raise ValueError(f"{raw['id']}: expected pdf or pages")
    if not require_graph:
        return Document(raw["id"], pages, source_pdf=pdf)
    if "layout" in raw:
        return read_mineru(raw["id"], pages, root / raw["layout"], pdf)
    macros, mentions = [], []
    for node in raw.get("macro", []):
        node = node.copy()
        box = node.pop("bbox", None)
        node["region"] = Region(node["page"], tuple(box)) if box else None
        node["references"] = tuple(node.get("references", ()))
        macros.append(MacroNode(**node))
    for node in raw.get("mentions", []):
        node = node.copy()
        box = node.pop("bbox", None)
        node["region"] = Region(node["page"], tuple(box)) if box else None
        mentions.append(Mention(**node))
    return Document(raw["id"], pages, tuple(macros), tuple(mentions), pdf)
