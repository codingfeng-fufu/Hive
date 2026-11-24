import asyncio
import base64
import io
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

from src.document import Document
from src.generation import _image
from src.retriever import Evidence, Retrieval

_IMAGE_PATH = re.compile(
    r"Image Path:\s*((?:(?!Image Path:)[^\r\n])*?\.(?:png|jpe?g|webp|tiff?|bmp))"
    r'(?=[.,;:)\]]?(?:\\[nrt"]|["\s]|$))',
    re.IGNORECASE,
)


class RAGAnything:
    def __init__(self, *, server: dict, working_dir: Path) -> None:
        from lightrag.utils import EmbeddingFunc
        from openai import AsyncOpenAI

        self.working_dir = working_dir.resolve()
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.llm = AsyncOpenAI(
            base_url=server["base_url"], api_key=os.getenv("VLLM_API_KEY", "local")
        )
        self.model = server["model"]
        self.completion_limit = int(os.getenv("HIVG_RAGANYTHING_COMPLETION_LIMIT", "2048"))
        if self.completion_limit < 1:
            raise ValueError("HIVG_RAGANYTHING_COMPLETION_LIMIT must be positive")
        self.loop = asyncio.new_event_loop()
        self.active_index = None
        embedding_url = os.environ["HIVG_GRAPH_EMBED_URL"]
        embedding_model = os.environ["HIVG_GRAPH_EMBED_MODEL"]
        self.embedding_client = AsyncOpenAI(
            base_url=embedding_url,
            api_key=os.getenv("HIVG_GRAPH_EMBED_API_KEY", "local"),
        )

        async def embed(texts: list[str], **kwargs):
            import numpy as np

            result = await self.embedding_client.embeddings.create(
                model=embedding_model, input=texts
            )
            return np.asarray(
                [item.embedding for item in sorted(result.data, key=lambda v: v.index)]
            )

        self.embedding = EmbeddingFunc(
            embedding_dim=int(os.environ["HIVG_GRAPH_EMBED_DIM"]),
            max_token_size=int(os.getenv("HIVG_GRAPH_EMBED_MAX_TOKENS", "8192")),
            func=embed,
        )

    async def _complete(
        self,
        prompt: str = "",
        system_prompt: str | None = None,
        history_messages: list[dict] | None = None,
        image_data: str | None = None,
        messages: list[dict] | None = None,
        **kwargs,
    ) -> str:
        if messages is None:
            messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
            content = prompt
            if image_data is not None:
                with Image.open(io.BytesIO(base64.b64decode(image_data))) as image:
                    content = [{"type": "text", "text": prompt}, _image(image, None, "")]
            messages += [*(history_messages or []), {"role": "user", "content": content}]
        requested = kwargs.get("max_tokens") or self.completion_limit
        optional = (
            {"response_format": kwargs["response_format"]} if "response_format" in kwargs else {}
        )
        response = await self.llm.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            max_tokens=min(requested, self.completion_limit),
            **optional,
        )
        return response.choices[0].message.content or ""

    def index(self, document: Document) -> tuple:
        from raganything import RAGAnything as Upstream
        from raganything.config import RAGAnythingConfig

        if self.active_index is not None:
            self.loop.run_until_complete(self._close_index())
        directory = Path(tempfile.mkdtemp(dir=self.working_dir))
        pdf = document.source_pdf
        if pdf is None:
            import fitz

            pdf = directory / "document.pdf"
            with fitz.open() as output:
                for _, image in sorted(document.pages.items()):
                    with fitz.open(image) as source:
                        output.insert_pdf(fitz.open("pdf", source.convert_to_pdf()))
                output.save(pdf)
        parse_dir = directory / "parsed"

        # LightRAG copies its configuration. A bound method would copy the API clients.
        async def complete(*args, **kwargs) -> str:
            return await self._complete(*args, **kwargs)

        rag = Upstream(
            config=RAGAnythingConfig(
                working_dir=str(directory / "graph"),
                parser="mineru",
                parser_output_dir=str(parse_dir),
            ),
            llm_model_func=complete,
            vision_model_func=complete,
            embedding_func=self.embedding,
            # LightRAG also keys its in-process stores by workspace.
            lightrag_kwargs={"workspace": directory.name},
        )
        self.active_index = rag
        layout_root = os.getenv("HIVG_RAGANYTHING_LAYOUT_ROOT")
        layouts = (
            list(Path(layout_root).glob(f"*/{pdf.stem}/vlm/{pdf.stem}_content_list.json"))
            if layout_root
            else []
        )
        if len(layouts) > 1:
            raise ValueError(f"Multiple precomputed MinerU layouts for {pdf.name}")
        if layouts:
            parse_dir.mkdir(parents=True, exist_ok=True)
            layout = layouts[0]
            content = json.loads(layout.read_text())
            for block in content:
                if asset := block.get("img_path"):
                    original = layout.parent / asset
                    if original.is_file():
                        staged = parse_dir / original.name
                        shutil.copyfile(original, staged)
                        block["img_path"] = str(staged)
            self.loop.run_until_complete(rag.insert_content_list(content, file_path=str(pdf)))
        else:
            self.loop.run_until_complete(
                rag.process_document_complete(str(pdf), output_dir=str(parse_dir))
            )
        return rag, parse_dir

    def retrieve(self, index: tuple, question: str) -> Retrieval:
        rag, parse_dir = index
        context = self.loop.run_until_complete(
            rag.aquery(question, mode="mix", only_need_context=True, vlm_enhanced=False),
        )
        if not isinstance(context, str):
            raise TypeError("Expected the upstream RAG-Anything context to be a string")
        crops: dict[Path, str] = {}
        for match in _IMAGE_PATH.finditer(context):
            location = match.group(1).strip()
            if urlparse(location).scheme:
                continue
            crop = Path(location).resolve()
            if crop.is_relative_to(parse_dir.resolve()) and crop.is_file():
                crops.setdefault(crop, f"Image {len(crops) + 1}")
        for path, label in crops.items():
            context = context.replace(str(path), label)
        evidence = [Evidence("context", "context", None, context, include_image=False)]
        evidence.extend(
            Evidence(label, label, None, image_path=crop) for crop, label in crops.items()
        )
        return Retrieval(evidence, {"mode": "rag-anything", "retrieved_images": len(crops)})

    async def _close_index(self) -> None:
        rag = self.active_index
        self.active_index = None
        try:
            await rag.finalize_storages()
        finally:
            graph = rag.lightrag
            if graph is not None:
                embedding = getattr(graph.embedding_func, "func", None)
                for callback in (graph.llm_model_func, embedding):
                    if shutdown := getattr(callback, "shutdown", None):
                        await shutdown()

    def close(self) -> None:
        async def close_clients() -> None:
            try:
                if self.active_index is not None:
                    await self._close_index()
            finally:
                await asyncio.gather(self.llm.close(), self.embedding_client.close())

        try:
            self.loop.run_until_complete(close_clients())
        finally:
            self.loop.close()
