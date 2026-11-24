"""Score page-level grounding from MinerU-derived names before comparing to annotations."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import fitz
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from eval.vger_pilot import _crop, _label, _overlap

PAGE_BOX = (0, 0, 1000, 1000)


def _predict(processor, model, image, phrases, device):
    batch = processor(
        images=[image] * len(phrases),
        text=[phrase.lower().rstrip(".") + "." for phrase in phrases],
        return_tensors="pt",
        padding=True,
    ).to(device)
    with torch.inference_mode():
        result = model(**batch)
    detections = processor.post_process_grounded_object_detection(
        result,
        batch.input_ids,
        threshold=0.05,
        text_threshold=0.20,
        target_sizes=[(1000, 1000)] * len(phrases),
    )
    return [
        [
            tuple(min(1000.0, max(0.0, x)) for x in box)
            for box in detection["boxes"][detection["scores"].argsort(descending=True)].tolist()
        ]
        for detection in detections
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--pdfs", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = perf_counter()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    candidates = json.loads(args.candidates.read_text())
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).to(args.device).eval()
    boxes = {}
    processed = 0
    for doc, pages in candidates.items():
        with fitz.open(args.pdfs / f"{doc}.pdf") as pdf:
            for page, options in pages.items():
                page = int(page)
                phrases = defaultdict(set)
                for label, prompt in options.items():
                    if prompt and _label(label):
                        phrases[prompt].add(_label(label))
                if not phrases:
                    continue
                image = _crop(pdf, page, PAGE_BOX, dpi=180)
                prompts = list(phrases)
                for i in range(0, len(prompts), args.batch_size):
                    group = prompts[i : i + args.batch_size]
                    detections = _predict(processor, model, image, group, args.device)
                    for prompt, detected in zip(group, detections, strict=True):
                        for label in phrases[prompt]:
                            boxes.setdefault((doc, page, label), []).extend(detected)
                    processed += len(group)
        print(doc, processed, flush=True)
    labels = json.loads(args.labels.read_text())["rows"]
    rows = []
    for anchor in labels:
        predicted = boxes.get((anchor["doc"], anchor["page"], _label(anchor["label"])), [])
        target = anchor["bbox"]
        ious = [_overlap(target, box) for box in predicted if box[0] < box[2] and box[1] < box[3]]
        rows.append(
            {
                **anchor,
                "automatic_prompt": (anchor["doc"], anchor["page"], _label(anchor["label"]))
                in boxes,
                "predicted_boxes": predicted[:5],
                "top1_iou": max(ious[:1], default=0),
                "top5_iou": max(ious[:5], default=0),
            }
        )
    result = {}
    for split in ("development", "heldout", "all"):
        selected = [row for row in rows if split == "all" or row["split"] == split]
        result[split] = {
            "n": len(selected),
            "prompt_coverage": sum(row["automatic_prompt"] for row in selected),
            "top1_iou_0_5": sum(row["top1_iou"] >= 0.5 for row in selected),
            "top5_iou_0_5": sum(row["top5_iou"] >= 0.5 for row in selected),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "model": args.model,
                "device": args.device,
                "prompts": processed,
                "seconds": perf_counter() - started,
                "summary": result,
                "rows": rows,
            },
            indent=2,
        )
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
