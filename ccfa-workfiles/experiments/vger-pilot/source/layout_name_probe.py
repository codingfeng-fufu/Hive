"""Generate candidate labels from actual MinerU lists, cells, captions, and short text."""

import argparse
import json
import re
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path

from eval.vger_pilot import _label

NUMBERED = re.compile(r"^\s*(?:\((\d{1,2}|[A-Da-d])\)|(\d{1,2}))[.)\s:]+(?=[A-Za-z])")
FIGURE_PREFIX = re.compile(r"^(fig(?:ure)?|table)\s*[. ]?\s*(\d+(?:[.-]\d+)*[a-z]?)", re.IGNORECASE)


class Cells(HTMLParser):
    def __init__(self, body: str) -> None:
        super().__init__()
        self.active = False
        self.segments = []
        self.cells = []
        self.feed(body)

    def handle_starttag(self, tag, attrs):
        if tag in {"th", "td"}:
            self.active = True
        if tag == "br" and self.active:
            self.segments.append("\n")

    def handle_data(self, data):
        if self.active:
            self.segments.append(data)

    def handle_endtag(self, tag):
        if tag in {"th", "td"}:
            self.cells.append("".join(self.segments))
            self.segments.clear()
            self.active = False


def _segments(block: dict) -> list[str]:
    parts = [block.get("text", ""), *block.get("list_items", [])]
    for key in ("image_caption", "table_caption"):
        parts.extend(block.get(key, []))
    if block.get("table_body"):
        parts.extend(Cells(block["table_body"]).cells)
    return [line.strip() for value in parts for line in str(value).splitlines() if line.strip()]


def names_from_layout(blocks: list[dict]) -> dict[int, dict[str, str | None]]:
    names = defaultdict(dict)
    for block in blocks:
        if block["type"] not in {"text", "list", "image", "table", "chart"}:
            continue
        page = block["page_idx"] + 1
        for line in _segments(block):
            if match := FIGURE_PREFIX.match(line):
                prefix = "Figure" if match[1].lower().startswith("fig") else "Table"
                names[page][f"{prefix} {match[2]}"] = None
                remainder = line[match.end() :].strip(" .:-")
                if 1 <= len(remainder.split()) <= 8:
                    names[page][remainder] = remainder
            numbered = NUMBERED.match(line)
            if numbered:
                explanation = line[numbered.end() :]
                short_description = " ".join(explanation.strip(" :;.-").split()[:8])
                identifier = numbered[1] or numbered[2]
                names[page][identifier] = None
                for endpoint in re.split(r"\s*\(|\s+for\b|\s+with\b|[.:;]", explanation, 1):
                    if 1 <= len(endpoint.split()) <= 8:
                        names[page][endpoint.strip()] = short_description
                continue
            if 1 <= len(line.split()) <= 8 and any(c.isalpha() for c in line):
                names[page][line.strip(" .:;")] = line.strip(" .:;")
    return dict(names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout-dir", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = {
        path.stem: names_from_layout(json.loads(path.read_text()))
        for path in sorted(args.layout_dir.glob("*.json"))
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(args.scores.read_text())["rows"]
    endpoints = {
        (
            row["document"],
            row["split"],
            row[f"{side}_figure"],
            row[f"{side}_page"],
            row[f"{side}_label"],
            tuple(row[f"{side}_bbox"]),
        )
        for row in rows
        for side in ("left", "right")
    }
    hits = []
    for doc, split, fig, page, label, box in sorted(endpoints):
        options = candidates.get(doc, {}).get(page, {})
        keys = {normalized for candidate in options if (normalized := _label(candidate))}
        hits.append(
            {
                "doc": doc,
                "split": split,
                "figure": fig,
                "page": page,
                "label": label,
                "bbox": box,
                "hit": bool(_label(label) and _label(label) in keys),
            }
        )
    totals = {}
    for split in ("development", "heldout", "all"):
        selection = [row for row in hits if split == "all" or row["split"] == split]
        totals[split] = {"n": len(selection), "hits": sum(row["hit"] for row in selection)}
    (args.output / "layout-name-candidates.json").write_text(json.dumps(candidates, indent=2))
    (args.output / "layout-name-probe.json").write_text(
        json.dumps(
            {
                "summary": totals,
                "rows": hits,
            },
            indent=2,
        )
    )
    print(totals)


if __name__ == "__main__":
    main()
