import base64
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image, ImageOps

from .document import Document
from .retriever import Evidence


@dataclass(frozen=True)
class Answer:
    content: str
    prompt_tokens: int | None
    output_tokens: int | None
    finish_reason: str | None


class VLLM:
    def __init__(self, base_url: str, model: str, max_tokens: int, timeout: float) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout

    def generate(
        self,
        document: Document,
        question: str,
        evidence: list[Evidence],
        trace_dir: Path | None = None,
        *,
        seed: int = 0,
        query_image: Image.Image | None = None,
    ) -> Answer:
        parts = []
        for i, item in enumerate(evidence):
            text = f"Page {item.page + 1}. {item.text}" if item.page is not None else item.text
            if text:
                parts.append({"type": "text", "text": text})
            if item.include_image and item.image_path is not None:
                with Image.open(item.image_path) as image:
                    parts.append(_image(image, trace_dir, f"evidence-{i:02}.png"))
            elif item.include_image and item.region:
                parts.append(_image(document.crop(item.region), trace_dir, f"evidence-{i:02}.png"))
            elif item.include_image and not item.text and item.page in document.pages:
                with Image.open(document.pages[item.page]) as page:
                    parts.append(_image(page, trace_dir, f"evidence-{i:02}.png"))
        if query_image is not None:
            parts.append({"type": "text", "text": "Query image"})
            parts.append(_image(query_image, trace_dir, "query.png"))
        parts.append({"type": "text", "text": f"Question: {question}\nShort answer:"})
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer the question using the supplied document evidence. "
                        "Return the shortest complete answer without explanation. "
                        "If the evidence cannot answer the question, return Not answerable."
                    ),
                },
                {"role": "user", "content": parts},
            ],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "seed": seed,
        }
        headers = {"Content-Type": "application/json"}
        if os.environ.get("VLLM_API_KEY"):
            headers["Authorization"] = f"Bearer {os.environ['VLLM_API_KEY']}"
        request = Request(self.url, data=json.dumps(data).encode(), headers=headers)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as error:
            raise RuntimeError(f"vLLM HTTP {error.code}: {error.read().decode()[:1000]}") from error
        choice = result["choices"][0]
        usage = result.get("usage") or {}
        return Answer(
            choice["message"].get("content") or "",
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            choice.get("finish_reason"),
        )


def _image(image: Image.Image, trace_dir: Path | None, filename: str) -> dict:
    # Qwen2.5-VL rejects images shorter than a 28-pixel patch on either side.
    # Pad unusual crops at upload time to preserve the original pixels and BBox.
    width, height = image.size
    minimum_edge = max(28, (max(width, height) + 199) // 200)
    if min(width, height) < minimum_edge:
        pad_width = max(0, minimum_edge - width)
        pad_height = max(0, minimum_edge - height)
        image = ImageOps.expand(
            image.convert("RGB"),
            (pad_width // 2, pad_height // 2, (pad_width + 1) // 2, (pad_height + 1) // 2),
            fill="white",
        )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    data = buffer.getvalue()
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
        (trace_dir / filename).write_bytes(data)
    return {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode("ascii")},
    }
