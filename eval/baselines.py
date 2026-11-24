import os
from pathlib import Path
from typing import Protocol

from PIL import Image

from src.document import Document
from src.retriever import Evidence, Retrieval


class Baseline(Protocol):
    def index(self, document: Document) -> object: ...
    def retrieve(self, index: object, question: str) -> Retrieval: ...


def _page_results(pages: list[int], scores: list[float], limit: int, method: str) -> Retrieval:
    ranked = sorted(zip(pages, scores), key=lambda item: (-item[1], item[0]))[:limit]
    return Retrieval(
        [Evidence(f"page-{page}", f"page-{page}", page, score=score) for page, score in ranked],
        {"mode": method, "selected_pages": [page for page, _ in ranked]},
    )


def _images(paths: list[Path]) -> list[Image.Image]:
    images = []
    for path in paths:
        with Image.open(path) as page:
            images.append(page.convert("RGB"))
    return images


class ColPali:
    def __init__(self, k: int = 6, batch_size: int = 2) -> None:
        import torch

        self.device = os.environ.get("HIVG_BASELINE_DEVICE", "cuda")
        model = os.environ.get("HIVG_COLPALI_MODEL", "vidore/colpali-v1.3")
        self.transformers_checkpoint = os.getenv("HIVG_COLPALI_FORMAT") == "transformers"
        if self.transformers_checkpoint:
            from transformers import ColPaliForRetrieval as ColPaliModel
            from transformers import ColPaliProcessor
        else:
            from colpali_engine.models import ColPali as ColPaliModel
            from colpali_engine.models import ColPaliProcessor

        self.processor = ColPaliProcessor.from_pretrained(model)
        # Transformers 4.51 Gemma promotes hidden states to float32 on this checkpoint.
        dtype = (
            torch.bfloat16
            if self.device.startswith("cuda") and not self.transformers_checkpoint
            else torch.float32
        )
        options = {"attn_implementation": "eager"} if self.transformers_checkpoint else {}
        self.model = (
            ColPaliModel.from_pretrained(model, torch_dtype=dtype, **options).to(self.device).eval()
        )
        self.k = k
        self.batch_size = batch_size
        self.torch = torch

    def index(self, document: Document) -> tuple[list[int], list]:
        pages = sorted(document.pages)
        features = []
        with self.torch.inference_mode():
            for start in range(0, len(pages), self.batch_size):
                images = _images(
                    [document.pages[page] for page in pages[start : start + self.batch_size]]
                )
                batch = self.processor.process_images(images).to(self.device)
                output = self.model(**batch)
                embeddings = output.embeddings if self.transformers_checkpoint else output
                features.extend(embeddings.cpu().unbind(0))
        return pages, features

    def retrieve(self, index: tuple[list[int], list], question: str) -> Retrieval:
        pages, features = index
        with self.torch.inference_mode():
            batch = self.processor.process_queries([question]).to(self.device)
            output = self.model(**batch)
            if self.transformers_checkpoint:
                scores = self.processor.score_retrieval(output.embeddings.cpu(), features)[0]
            else:
                embedding = output.cpu().unbind(0)
                scores = self.processor.score_multi_vector(embedding, features, device=self.device)[
                    0
                ]
        return _page_results(pages, scores.float().tolist(), self.k, "colpali")


class VisRAG:
    def __init__(self, k: int = 6, batch_size: int = 2) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = os.environ.get("HIVG_BASELINE_DEVICE", "cuda")
        name = os.environ.get("HIVG_VISRAG_MODEL", "openbmb/VisRAG-Ret")
        revision = os.environ.get("HIVG_VISRAG_REVISION")
        self.tokenizer = AutoTokenizer.from_pretrained(
            name, revision=revision, trust_remote_code=True
        )
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.model = (
            AutoModel.from_pretrained(
                name,
                revision=revision,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            .to(self.device)
            .eval()
        )
        self.k = k
        self.batch_size = batch_size
        self.torch = torch

    def _encode(self, text: list[str], images: list) -> object:
        with self.torch.inference_mode():
            output = self.model(text=text, image=images, tokenizer=self.tokenizer)
            mask = output.attention_mask
            weights = (mask * mask.cumsum(dim=1)).unsqueeze(-1).float()
            pooled = (output.last_hidden_state * weights).sum(dim=1) / weights.sum(dim=1)
            return self.torch.nn.functional.normalize(pooled, dim=1).cpu()

    def index(self, document: Document) -> tuple[list[int], object]:
        pages = sorted(document.pages)
        embeddings = []
        for start in range(0, len(pages), self.batch_size):
            batch = pages[start : start + self.batch_size]
            images = _images([document.pages[page] for page in batch])
            embeddings.append(self._encode([""] * len(images), images))
        return pages, self.torch.cat(embeddings)

    def retrieve(self, index: tuple[list[int], object], question: str) -> Retrieval:
        pages, embeddings = index
        query = self._encode(
            [f"Represent this query for retrieving relevant documents: {question}"], [None]
        )
        return _page_results(pages, (query @ embeddings.T)[0].tolist(), self.k, "visrag")
