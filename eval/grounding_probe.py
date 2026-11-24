"""Measure Qwen localization with an annotated figure and entity prompt."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import fitz
import torch
import yaml
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .vger_pilot import _crop, _overlap

PROMPT = (
    'Locate the entity "{label}" in the provided figure. The description of the figure '
    'is "{caption}". If the entity is a numeric callout, locate the object it points to. '
    "Return the tight rectangle as a JSON array of four integers, "
    "[x1,y1,x2,y2], in normalized coordinates from 0 to 1000. "
    "Return [] if the figure does not show the object. Output only the array."
)


def _box(text: str, figure: list[int]) -> tuple | None:
    match = re.search(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", text)
    if match is None:
        return None
    x1, y1, x2, y2 = map(int, match.groups())
    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
        return None
    left, top, right, bottom = figure
    return (
        left + (right - left) * x1 / 1000,
        top + (bottom - top) * y1 / 1000,
        left + (right - left) * x2 / 1000,
        top + (bottom - top) * y2 / 1000,
    )


def _infer(model, processor, image: Image.Image, question: str, device: str) -> tuple[str, int]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    batch = processor(text=[prompt], images=[image], return_tensors="pt").to(device)
    tokens = batch.input_ids.shape[1]
    with torch.inference_mode():
        generated = model.generate(
            **batch,
            do_sample=False,
            max_new_tokens=36,
            temperature=None,
            top_p=None,
        )
    answer = processor.decode(generated[0, tokens:], skip_special_tokens=True)
    return answer, tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--pdfs", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--per-document", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    annotations = yaml.safe_load(args.annotations.read_text())["documents"]
    processor = AutoProcessor.from_pretrained(args.model, min_pixels=56**2, max_pixels=448**2)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": args.device},
        attn_implementation="sdpa",
    ).eval()
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".jsonl").open("w") as stream:
        for name, document in annotations.items():
            if document["split"] != "heldout":
                continue
            with fitz.open(args.pdfs / f"{name}.pdf") as pdf:
                selected = document["regions"][: args.per_document]
                for fig, target, entity, labels in selected:
                    page, figure_box, caption = document["figures"][fig]
                    image = _crop(pdf, page, tuple(figure_box), dpi=180)
                    question = PROMPT.format(label=labels[-1], caption=caption)
                    start = perf_counter()
                    reply, tokens = _infer(model, processor, image, question, args.device)
                    predicted = _box(reply, figure_box)
                    row = {
                        "doc": name,
                        "entity": entity,
                        "label": labels[-1],
                        "page": page,
                        "figure": fig,
                        "target": target,
                        "predicted": predicted,
                        "reply": reply,
                        "iou": _overlap(tuple(target), predicted) if predicted else 0.0,
                        "prompt_tokens": tokens,
                        "seconds": perf_counter() - start,
                    }
                    rows.append(row)
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    print(name, page, entity, f"{row['iou']:.3f}", flush=True)
    by_doc = defaultdict(list)
    for row in rows:
        by_doc[row["doc"]].append(row["iou"])
    summary = {
        "model": args.model,
        "device": args.device,
        "per_document": args.per_document,
        "prompt": PROMPT,
        "samples": len(rows),
        "valid_boxes": sum(r["predicted"] is not None for r in rows),
        "hits_iou_0_5": sum(r["iou"] >= 0.5 for r in rows),
        "hits_iou_0_25": sum(r["iou"] >= 0.25 for r in rows),
        "per_doc": {
            doc: {"n": len(values), "hits_iou_0_5": sum(v >= 0.5 for v in values)}
            for doc, values in by_doc.items()
        },
        "seconds": sum(row["seconds"] for row in rows),
    }
    args.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
