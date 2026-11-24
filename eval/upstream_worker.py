"""Run the upstream MMGraphRAG or MG²-RAG in its own Python environment."""

import asyncio
import contextlib
import csv
import io
import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path


def _pdf_for(document: dict, directory: Path) -> Path:
    if document["pdf"]:
        return Path(document["pdf"])
    import fitz

    path = directory / "document.pdf"
    with fitz.open() as pdf:
        for _, raster in sorted(document["pages"].items(), key=lambda item: int(item[0])):
            with fitz.open(raster) as image, fitz.open("pdf", image.convert_to_pdf()) as page:
                pdf.insert_pdf(page)
        pdf.save(path)
    return path


class MMGraphWorker:
    def __init__(self, server: dict, workdir: Path) -> None:
        from src import parameter

        token = os.getenv("VLLM_API_KEY", "local")
        parameter.API_KEY = parameter.MM_API_KEY = token
        parameter.API_BASE = parameter.MM_API_BASE = server["base_url"]
        parameter.MODEL_NAME = parameter.MM_MODEL_NAME = server["model"]
        cache = workdir / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        parameter.CACHE_PATH = str(cache)
        parameter.MMKG_NAME = "document"
        self.parameter = parameter
        self.active = None

    def index(self, document: dict, directory: Path) -> None:
        from src.builder import MMKGBuilder
        from src.retrieval.query import GraphRAGQuery

        self.parameter.WORKING_DIR = str(directory / "working")
        self.parameter.OUTPUT_DIR = str(directory / "graph")
        pdf = _pdf_for(document, directory)
        if layout_root := os.getenv("HIVG_MMGRAPH_LAYOUT_ROOT"):
            layouts = sorted(Path(layout_root).glob(f"*/{pdf.stem}/vlm/{pdf.stem}.md"))
            if len(layouts) != 1:
                raise ValueError(f"Expected one cached MinerU Markdown layout for {pdf.name}")
            cached_path = Path(self.parameter.WORKING_DIR) / pdf.stem / "auto"
            cached_path.parent.mkdir(parents=True, exist_ok=True)
            cached_path.symlink_to(layouts[0].parent.resolve(), target_is_directory=True)
        MMKGBuilder(working_dir=self.parameter.WORKING_DIR).index(str(pdf))
        if not (Path(self.parameter.OUTPUT_DIR) / "document.graphml").is_file():
            raise RuntimeError("MMGraphRAG produced no document.graphml")
        self.active = GraphRAGQuery(
            working_dir=self.parameter.WORKING_DIR,
            output_dir=self.parameter.OUTPUT_DIR,
        )

    def retrieve(self, question: str, directory: Path, query_image: str | None) -> dict:
        from src.retrieval.query import QueryParam, _build_local_query_context

        rag = self.active
        config = QueryParam()
        entities, context = asyncio.run(
            _build_local_query_context(
                question, rag.graph, rag.embeddings, rag.text_chunks, config
            ),
        )
        evidence = []
        if not context:
            return {"evidence": evidence, "trace": {"mode": "mmgraphrag"}}
        evidence.append(
            {
                "node_id": "context",
                "occurrence_id": "context",
                "page": None,
                "text": context,
                "include_image": False,
            }
        )
        rows = csv.DictReader(io.StringIO(entities), skipinitialspace=True)
        images = list(
            dict.fromkeys(row["entity"].lower() for row in rows if "ORI_IMG" in row.get("type", ""))
        )[: config.number_of_mmentities]
        for i, entity in enumerate(images):
            metadata = rag.image_data.get(entity)
            if not metadata:
                continue
            path = Path(metadata["image_path"])
            if not path.is_file():
                path = Path(rag.working_dir) / "images" / path.name
            if path.is_file():
                asset = directory / f"retrieved-{i}{path.suffix}"
                if path.resolve() != asset.resolve():
                    shutil.copyfile(path, asset)
                evidence.append(
                    {
                        "node_id": entity,
                        "occurrence_id": entity,
                        "page": None,
                        "text": metadata.get("caption") or "",
                        "image_path": str(asset),
                    }
                )
        return {"evidence": evidence, "trace": {"mode": "mmgraphrag"}}


class MG2Worker:
    def __init__(self, server: dict, workdir: Path) -> None:
        from dotenv import load_dotenv

        load_dotenv(Path.cwd() / ".env")
        self.server = server
        self.active = None
        self.evidence = {}

    def index(self, document: dict, directory: Path) -> None:
        import fitz
        from src.mmgraphrag.MMGraphRAG import MMGraphRAG
        from src.mmgraphrag.utils.config_utils import BaseConfig

        pages = sorted((int(page), path) for page, path in document["pages"].items())
        with fitz.open(_pdf_for(document, directory)) as pdf:
            docs = [f"Page {page + 1}\n{pdf[page].get_text()}" for page, _ in pages]
        if self.active is not None:
            self.active = None
        config = BaseConfig.from_env(save_dir=str(directory / "graph"))
        config.mllm_name = self.server["model"]
        config.mllm_base_url = self.server["base_url"]
        rag = MMGraphRAG(global_config=config)
        rag.index(docs=docs, imgs=[path for _, path in pages], match_arr=list(range(len(docs))))
        self.active = rag
        self.evidence = {text: page for text, (page, _) in zip(docs, pages)}

    def retrieve(self, question: str, directory: Path, query_image: str | None) -> dict:
        (solution,) = self.active.retrieve(question_queries=[question], image_queries=[query_image])
        evidence = []
        limit = int(os.getenv("HIVG_MG2_TOP_K", str(self.active.global_config.qa_top_k)))
        for text, score in list(zip(solution.docs, solution.doc_scores))[:limit]:
            page = self.evidence[text]
            evidence.extend(
                [
                    {
                        "node_id": f"page-{page}",
                        "occurrence_id": f"page-{page}:text",
                        "page": page,
                        "text": text,
                        "score": float(score),
                        "include_image": False,
                    },
                    {
                        "node_id": f"page-{page}",
                        "occurrence_id": f"page-{page}:image",
                        "page": page,
                        "score": float(score),
                    },
                ]
            )
        return {"evidence": evidence, "trace": {"mode": "mg2-rag"}}


def main() -> None:
    method, clone = sys.argv[1:]
    os.chdir(Path(clone).resolve())
    sys.path.insert(0, str(Path.cwd()))
    worker = None
    active = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            workdir = Path(request["workdir"])
            with contextlib.redirect_stdout(sys.stderr):
                if worker is None:
                    factory = {"mmgraphrag": MMGraphWorker, "mg2-rag": MG2Worker}[method]
                    worker = factory(request["server"], workdir)
                if request["operation"] == "index":
                    active = Path(tempfile.mkdtemp(dir=workdir))
                    worker.index(request["document"], active)
                    response = {"index": str(active)}
                elif request["operation"] == "retrieve" and active == Path(request["index"]):
                    response = worker.retrieve(
                        request["question"],
                        active,
                        request.get("query_image"),
                    )
                else:
                    raise ValueError("The worker requires an active graph index")
        except Exception as error:  # noqa: BLE001 - preserve the worker and report the upstream failure
            traceback.print_exc(file=sys.stderr)
            response = {"error": f"{type(error).__name__}: {error}"}
        print(json.dumps(response, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
