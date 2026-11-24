from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .config import EncoderConfig

Vector = NDArray[np.float32]


def normalize(vectors: Vector) -> Vector:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norms > 0, norms, 1)


class Encoder(Protocol):
    def text(self, texts: Sequence[str]) -> Vector: ...
    def image(self, images: Sequence[Image.Image]) -> Vector: ...


class SiglipEncoder:
    def __init__(self, config: EncoderConfig) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.device = config.device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = config.batch_size
        self.processor = AutoProcessor.from_pretrained(
            config.model, revision=config.revision, use_fast=False
        )
        self.model = AutoModel.from_pretrained(config.model, revision=config.revision)
        self.model.to(self.device).eval()
        self.revision = getattr(self.model.config, "_commit_hash", None) or config.revision

    def text(self, texts: Sequence[str]) -> Vector:
        return self._encode("text", texts)

    def image(self, images: Sequence[Image.Image]) -> Vector:
        return self._encode("images", images)

    def _encode(self, modality: str, data: Sequence) -> Vector:
        if not data:
            raise ValueError("Encoder requires at least one input")
        blocks = []
        encode = (
            self.model.get_text_features if modality == "text" else self.model.get_image_features
        )
        with self.torch.inference_mode():
            for start in range(0, len(data), self.batch_size):
                kwargs = {modality: list(data[start : start + self.batch_size])}
                if modality == "text":
                    kwargs.update(padding="max_length", truncation=True)
                batch = self.processor(return_tensors="pt", **kwargs).to(self.device)
                features = encode(**batch)
                if hasattr(features, "pooler_output"):
                    features = features.pooler_output
                blocks.append(features.float().cpu().numpy())
        return normalize(np.concatenate(blocks).astype(np.float32))
