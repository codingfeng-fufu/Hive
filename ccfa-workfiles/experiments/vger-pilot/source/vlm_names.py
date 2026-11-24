"""Extract visual entity names with a local Qwen endpoint."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from layout_name_probe import names_from_layout

from src.document import Document
from src.generation import VLLM
from src.retriever import Evidence

PROMPT = (
    "Identify distinct entities directly shown as illustrations, symbols, charts, product "
    "parts, or panels. Use visible labels and nearby descriptions when available. "
    'Return a compact JSON array of ["label", "visual name"] entries. Include the literal '
    "figure callout or panel label when you can associate it to an object, as well as the "
    "short name used in the page text. The visual name should describe the drawn object. "
    "Use an empty array if there is no visible figure content. No prose or markdown."
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--layout-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--max-workers", type=int, default=3)
    args = parser.parse_args()
    pages = json.loads(args.pages.read_text())
    generator = VLLM(args.base_url, "Qwen/Qwen2.5-VL-7B-Instruct", 640, 150)
    jobs = []
    for doc, indices in pages.items():
        blocks = json.loads((args.layout_dir / f"{doc}.json").read_text())
        candidates = names_from_layout(blocks)
        jobs.extend((doc, page, candidates.get(page, {})) for page in indices)

    def infer(job):
        doc, page, options = job
        raster = args.layout_dir / f"{doc}-{page}.png"
        document = Document(doc, {page - 1: raster})
        clues = ", ".join(options)[:1200]
        answer = generator.generate(
            document,
            f"{PROMPT}\nPossible labels extracted from the page: {clues}",
            [Evidence("page", "page", page - 1)],
        )
        decoder = json.JSONDecoder()
        names = {}
        for i, char in enumerate(answer.content):
            if char != "[":
                continue
            try:
                values, _ = decoder.raw_decode(answer.content[i:])
            except json.JSONDecodeError:
                continue
            for item in values:
                if (
                    isinstance(item, list)
                    and len(item) == 2
                    and all(isinstance(part, str) and part.strip() for part in item)
                ):
                    label, visual = (part.strip()[:100] for part in item)
                    names[label] = visual
                    names[visual] = visual
            break
        return doc, page, names, answer.content, answer.prompt_tokens

    args.output.mkdir(parents=True, exist_ok=True)
    candidates = {}
    with (
        ThreadPoolExecutor(max_workers=args.max_workers) as pool,
        (args.output / "vlm-name-replies.jsonl").open("w") as stream,
    ):
        for doc, page, names, answer, tokens in pool.map(infer, jobs):
            candidates.setdefault(doc, {})[page] = names
            stream.write(
                json.dumps(
                    {"doc": doc, "page": page, "reply": answer, "prompt_tokens": tokens},
                    ensure_ascii=False,
                )
                + "\n"
            )
            stream.flush()
            print(doc, page, len(names), flush=True)
    (args.output / "vlm-name-candidates.json").write_text(
        json.dumps(
            candidates,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
