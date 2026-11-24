"""Rank figure windows without using object coordinates."""

import json
from collections import defaultdict
from pathlib import Path

import fitz
import numpy as np
import yaml

from eval.vger_pilot import _crop, _overlap
from src.config import EncoderConfig
from src.embeddings import SiglipEncoder

notes = Path("ccfa-workfiles/experiments/vger-pilot/source/annotations.yaml")
assets = Path("data/upstream/mmlongbench-doc/documents")
out = Path("runs/vger-pilot")
annotations = yaml.safe_load(notes.read_text())["documents"]
encoder = SiglipEncoder(
    EncoderConfig(model="data/models/siglip-so400m", device="cuda", batch_size=3)
)


def windows(box):
    x0, y0, x1, y1 = box
    for width, height in ((0.25, 0.25), (0.35, 0.35), (0.45, 0.30), (0.30, 0.45), (0.55, 0.55)):
        for left in np.linspace(0.0, 1 - width, 4):
            for top in np.linspace(0.0, 1 - height, 4):
                yield tuple(
                    round(v)
                    for v in (
                        x0 + left * (x1 - x0),
                        y0 + top * (y1 - y0),
                        x0 + (left + width) * (x1 - x0),
                        y0 + (top + height) * (y1 - y0),
                    )
                )


rows = []
for name, doc in annotations.items():
    if doc["split"] != "heldout":
        continue
    figures = defaultdict(list)
    for key, target, id, labels in doc["regions"][:3]:
        figures[key].append((target, id, labels[-1]))
    with fitz.open(assets / f"{name}.pdf") as pdf:
        for key, targets in figures.items():
            page, figure, _ = doc["figures"][key]
            boxes = list(windows(figure))
            vectors = encoder.image([_crop(pdf, page, box, dpi=150) for box in boxes])
            text = encoder.text([key for _, _, key in targets])
            for (target, entity, label), scores in zip(targets, text @ vectors.T):
                order = np.argsort(-scores)
                rows.append(
                    {
                        "doc": name,
                        "page": page,
                        "entity": entity,
                        "label": label,
                        "top1_iou": _overlap(target, boxes[order[0]]),
                        "top5_iou": max(_overlap(target, boxes[idx]) for idx in order[:5]),
                        "pool_iou": max(_overlap(target, box) for box in boxes),
                    }
                )
        print(name, len(rows), flush=True)
summary = {
    field: sum(row[field] >= 0.5 for row in rows) for field in ("top1_iou", "top5_iou", "pool_iou")
}
summary["n"] = len(rows)
(out / "region-proposals.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
print(summary)
