import argparse
import json
from pathlib import Path
from time import perf_counter

import fitz
import torch
import yaml
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from eval.vger_pilot import _crop, _overlap

source = Path("ccfa-workfiles/experiments/vger-pilot/source/annotations.yaml")
assets = Path("data/upstream/mmlongbench-doc/documents")
out = Path("runs/vger-pilot")
args_parser = argparse.ArgumentParser()
args_parser.add_argument("--device", default="cuda")
args_parser.add_argument("--layout-dir", type=Path)
args_parser.add_argument("--whole-page", action="store_true")
args = args_parser.parse_args()
device = args.device
checkpoint = "data/models/grounding-dino-tiny"
processor = AutoProcessor.from_pretrained(checkpoint)
model = AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint).to(device).eval()
rows = []
started = perf_counter()
for name, doc in yaml.safe_load(source.read_text())["documents"].items():
    if doc["split"] != "heldout":
        continue
    with fitz.open(assets / f"{name}.pdf") as pdf:
        for key, target, entity, labels in doc["regions"][:3]:
            page, parent, _ = doc["figures"][key]
            label = labels[-1]
            annotated_parent = parent
            if args.layout_dir:
                blocks = json.loads((args.layout_dir / f"{name}.json").read_text())
                candidates = [
                    block["bbox"]
                    for block in blocks
                    if block["page_idx"] == page - 1
                    and block["type"] in {"image", "table", "chart"}
                    and block.get("bbox")
                ]
                parent = max(
                    candidates, key=lambda box: _overlap(annotated_parent, box), default=None
                )
                if parent is None or _overlap(annotated_parent, parent) < 0.1:
                    rows.append(
                        {
                            "doc": name,
                            "entity": entity,
                            "label": label,
                            "page": page,
                            "target": target,
                            "candidates": 0,
                            "top1_iou": 0,
                            "top5_iou": 0,
                            "pool_iou": 0,
                            "parent_iou": 0,
                        }
                    )
                    continue
            if args.whole_page:
                parent = [0, 0, 1000, 1000]
            image = _crop(pdf, page, tuple(parent), dpi=180)
            batch = processor(
                images=image, text=label.lower().rstrip(".") + ".", return_tensors="pt"
            ).to(device)
            with torch.inference_mode():
                result = model(**batch)
            result = processor.post_process_grounded_object_detection(
                result,
                batch.input_ids,
                threshold=0.05,
                text_threshold=0.20,
                target_sizes=[(1000, 1000)],
            )[0]
            x0, y0, x1, y1 = parent
            boxes = []
            for box in result["boxes"][result["scores"].argsort(descending=True)].tolist():
                left, top, right, bottom = [min(1000, max(0, value)) / 1000 for value in box]
                if left < right and top < bottom:
                    boxes.append(
                        (
                            x0 + left * (x1 - x0),
                            y0 + top * (y1 - y0),
                            x0 + right * (x1 - x0),
                            y0 + bottom * (y1 - y0),
                        )
                    )
            ious = [_overlap(target, box) for box in boxes]
            rows.append(
                {
                    "doc": name,
                    "entity": entity,
                    "label": label,
                    "page": page,
                    "target": target,
                    "candidates": len(boxes),
                    "parent_iou": _overlap(parent, annotated_parent),
                    "top1_iou": max(ious[:1], default=0),
                    "top5_iou": max(ious[:5], default=0),
                    "pool_iou": max(ious, default=0),
                }
            )
        print(name, len(rows), flush=True)
summary = {
    field: sum(row[field] >= 0.5 for row in rows) for field in ("top1_iou", "top5_iou", "pool_iou")
}
summary["n"] = len(rows)
summary["seconds"] = perf_counter() - started
if args.whole_page:
    filename = "grounding-dino-page.json"
elif args.layout_dir:
    filename = "grounding-dino-mineru-figure.json"
else:
    filename = "grounding-dino.json"
(out / filename).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
print(summary)
