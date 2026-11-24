"""Render a side-by-side case comparison from benchmark retrieval traces."""

import argparse
import json
import os
from collections import defaultdict
from html import escape
from pathlib import Path
from urllib.parse import quote

STYLE = """
body { max-width: 1440px; margin: 36px auto; padding: 0 24px; color: #17212b;
       font: 16px/1.5 'Source Sans 3', system-ui, sans-serif; background: #f2f5f7; }
section { margin-bottom: 32px; padding: 20px; background: white; border: 1px solid #ccd6de; }
h1, h2, h3, p { margin: 0 0 12px; }
h2 { font-size: 1.25rem; }
.columns { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }
article { border-top: 3px solid #147569; padding: 12px 8px; min-width: 0; }
details { margin: 12px 0; border: 1px solid #ccd6de; padding: 8px; }
summary { cursor: pointer; }
img { max-width: 100%; max-height: 240px; object-fit: contain; }
code { overflow-wrap: anywhere; }
"""


def _link(path: Path, output: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return escape(quote(os.path.relpath(path, output.parent)), quote=True)


def _column(path: Path, row: dict, output: Path) -> str:
    parts = [
        f"<article><h3>{escape(row['method'])}</h3>",
        f"<p><b>Answer</b> {escape(row['prediction'])}</p>",
        f"<p>ANLS {row['anls']:.3f} · {row['prompt_tokens']} prompt tokens</p>",
        f"<p>Route <b>{escape(row['mode'])}</b> · γ = {row['gamma']:.3f}</p>",
        "<p><b>Selected parents</b> "
        + (", ".join(f"<code>{escape(parent)}</code>" for parent in row["selected_parents"]) or "—")
        + "</p>",
    ]
    for item in row["evidence"]:
        page = "unknown" if item["page"] is None else str(item["page"] + 1)
        description = escape(item["occurrence_id"])
        parts.append(f"<details><summary>Page {page} · <code>{description}</code></summary>")
        if item["region"]:
            parts.append(f"<p>BBox <code>{item['region']['bbox']}</code></p>")
        if item["text"]:
            parts.append(f"<p>{escape(item['text'])}</p>")
        if image := item.get("image_file"):
            asset = _link(path.parent / image, output)
            parts.append(f'<a href="{asset}"><img src="{asset}" alt="Evidence on page {page}"></a>')
        parts.append("</details>")
    parts.append(
        f'<p><a href="{_link(path, output)}">Full trace, entity splits and aliases</a></p>'
    )
    parts.append("</article>")
    return "\n".join(parts)


def render(roots: list[Path], output: Path) -> None:
    cases = defaultdict(list)
    paths = sorted(path for root in roots for path in root.rglob("trace.json"))
    for path in paths:
        row = json.loads(path.read_text())
        cases[row["doc_id"], row["question_id"]].append((path, row))
    if not cases:
        raise ValueError(f"No traces under {roots}")
    content = [
        '<!doctype html><html lang="en"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>HiVG-RAG retrieval cases</title><style>{STYLE}</style>",
        "<h1>HiVG-RAG retrieval cases</h1>",
    ]
    order = {"hivg-rag": 0, "macro-only": 1, "micro-only": 2}
    for (document, identifier), group in sorted(cases.items()):
        row = group[0][1]
        answers = " / ".join(map(str, row["answers"]))
        content.extend(
            [
                f"<section><h2>{escape(row['question'])}</h2>",
                f"<p>{escape(document)} · question {escape(identifier)} · {escape(row['subset'])}</p>",
                f"<p><b>Reference</b> {escape(answers)}</p>",
                '<div class="columns">',
            ]
        )
        for path, result in sorted(group, key=lambda item: order.get(item[1]["method"], 3)):
            content.append(_column(path, result, output))
        content.append("</div></section>")
    content.append("</html>")
    output.write_text("\n".join(content) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--additional-root", type=Path, action="append", default=[])
    args = parser.parse_args()
    render([args.run, *args.additional_root], args.output)
