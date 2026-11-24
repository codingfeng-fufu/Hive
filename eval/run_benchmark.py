import argparse
import csv
import hashlib
import json
import re
import sys
from collections.abc import Callable
from dataclasses import asdict, replace
from functools import lru_cache
from importlib import import_module, metadata
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml
from PIL import Image

from src.config import Config, GraphConfig, load_config
from src.embeddings import Encoder, SiglipEncoder
from src.generation import VLLM
from src.graph_builder import GraphBuilder, GraphIndex
from src.layout import load_document
from src.retriever import Evidence, Retrieval, Retriever

from .baseline_command import CommandBaseline
from .checkpoint import DocumentCheckpoints, lock_output
from .metrics import score, summarize, topology
from .stress import LEVELS, inject, select_documents

VARIANTS = (
    "hivg-rag",
    "without-vger-split",
    "without-vger-bridge",
    "macro-only",
    "micro-only",
    "without-native-crop",
)
BASELINES = ("direct", "visrag", "colpali", "rag-anything", "mmgraphrag", "mg2-rag")
SPLIT_GRID = (0.4, 0.5, 0.6, 0.7, 0.8)
SLUG = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*")


def load_symbol(spec: str) -> Any:
    module, sep, symbol = spec.partition(":")
    if not sep:
        raise ValueError(f"Expected module:symbol, received {spec}")
    return getattr(import_module(module), symbol)


def _changes(config: Config, method: str) -> Config:
    if method == "without-vger-split":
        return replace(config, graph=replace(config.graph, enable_split=False))
    if method == "without-vger-bridge":
        return replace(config, graph=replace(config.graph, enable_bridge=False))
    if method == "without-native-crop":
        return replace(config, retrieval=replace(config.retrieval, native_crop=False))
    return config


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row if key != "trace"))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _environment() -> dict[str, str]:
    packages = {}
    for name in (
        "numpy",
        "networkx",
        "scipy",
        "Pillow",
        "PyMuPDF",
        "transformers",
        "torch",
        "datasets",
        "raganything",
        "colpali-engine",
    ):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return packages


def _case_name(doc_id: str, question_id: str) -> str:
    return hashlib.sha256(json.dumps([doc_id, question_id]).encode()).hexdigest()[:24]


def _direct_pages(pages: list[int], limit: int | None) -> list[int]:
    if limit is None or len(pages) <= limit:
        return pages
    if limit == 1:
        return pages[:1]
    return [pages[i * (len(pages) - 1) // (limit - 1)] for i in range(limit)]


def _load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    doc_ids = [document["id"] for document in manifest["documents"]]
    if not doc_ids or len(set(doc_ids)) != len(doc_ids):
        raise ValueError(f"{path}: expected unique document IDs")
    if not SLUG.fullmatch(manifest["dataset"]):
        raise ValueError(f"{path}: dataset name must be a path-safe slug")
    for document in manifest["documents"]:
        question_ids = [question["id"] for question in document["questions"]]
        if not question_ids or len(set(question_ids)) != len(question_ids):
            raise ValueError(f"{document['id']}: expected unique question IDs")
        for question in document["questions"]:
            if question.get("subset", "unlabeled") not in ("macro", "micro", "unlabeled"):
                raise ValueError(f"{question['id']}: subset must be macro, micro, or unlabeled")
            if not isinstance(question["answers"], list) or not question["answers"]:
                raise ValueError(f"{question['id']}: expected a nonempty answers list")
            if not question.get("question", "").strip():
                raise ValueError(f"{question['id']}: question is empty")
    return manifest


def _trace(index: GraphIndex | None, retrieved: Retrieval) -> dict:
    trace = dict(retrieved.trace)
    trace["evidence"] = [
        {
            **asdict(evidence),
            "image_path": str(evidence.image_path) if evidence.image_path else None,
        }
        for evidence in retrieved.evidence
    ]
    # Generation names the image by its evidence position, including positions with text only.
    for i, item in enumerate(trace["evidence"]):
        if item["include_image"] and (
            item["region"] or item["image_path"] or (item["page"] is not None and not item["text"])
        ):
            item["image_file"] = f"evidence-{i:02}.png"
    if index is not None:
        trace["resolved_mentions"] = index.entity_of
        trace["split_pairs"] = [
            {
                **asdict(pair),
                "left_entity": index.entity_of[pair.left],
                "right_entity": index.entity_of[pair.right],
            }
            for pair in index.splits
        ]
        trace["alias_edges"] = [asdict(alias) for alias in index.aliases]
    return trace


def _sensitivity_indices(
    entry: dict, config: Config, cache: Path, encoder: Encoder
) -> Callable[[str, GraphConfig], GraphIndex]:
    raw_by_id = {doc["id"]: doc for doc in entry["docs"]}

    @lru_cache(maxsize=32)
    def load(doc_id: str):
        return load_document(raw_by_id[doc_id], entry["path"].parent, cache, config.dpi)

    @lru_cache(maxsize=32)
    def baseline(doc_id: str) -> GraphIndex:
        return GraphBuilder(encoder, config.graph).build(load(doc_id))

    @lru_cache(maxsize=32)
    def index(doc_id: str, settings: GraphConfig) -> GraphIndex:
        encoded = baseline(doc_id)
        if settings == config.graph:
            return encoded
        return GraphBuilder(encoder, settings).build(
            encoded.document, features=(encoded.text, encoded.visual)
        )

    return index


def run(
    entry: dict,
    config: Config,
    method: str,
    tag: str,
    backbone: str,
    server: dict,
    output: Path,
    cache: Path,
    encoder: Encoder | None,
    *,
    trace: bool,
    distractors: int,
    stress: bool,
    seed: int,
    adapter_spec: str | dict | None,
    scorer_spec: str | None,
    case_filter: set[str] | None = None,
    direct_max_pages: int | None = None,
    sensitivity_index: Callable[[str, GraphConfig], GraphIndex] | None = None,
    resume: bool = False,
) -> dict:
    output.mkdir(parents=True, exist_ok=resume)
    manifest, path = entry["data"], entry["path"]
    definitions = (method, backbone, tag)
    case = dict(zip(("method", "backbone", "setting"), definitions))
    checkpoints = DocumentCheckpoints(
        output,
        {
            **case,
            "dataset": manifest["dataset"],
            "split": manifest.get("split"),
            "granularity_protocol": manifest.get("granularity_protocol"),
            "manifest": str(path),
            "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "config": asdict(config),
            "encoder_revision": getattr(encoder, "revision", None),
            "server": server,
            "seed": seed,
            "adapter": adapter_spec,
            "scorer": scorer_spec,
            "direct_max_pages": direct_max_pages if method == "direct" else None,
            "reuses_sensitivity_index": sensitivity_index is not None,
            "checkpoint_version": 1,
            "case_filter": sorted(case_filter) if case_filter is not None else None,
            "python": sys.version.split()[0],
            "packages": _environment(),
            "stress_documents": {
                doc["id"]: doc["distractors"][:distractors] for doc in entry["docs"]
            }
            if stress
            else None,
        },
        resume=resume,
    )
    generator = VLLM(
        server["base_url"],
        server["model"],
        server.get("max_tokens", 128),
        server.get("timeout", 180),
    )
    graph_mode = method in VARIANTS
    route = {"macro-only": "macro", "micro-only": "micro"}.get(method, "adaptive")
    before = replace(config.graph, enable_split=False, enable_bridge=False)
    adapter = None
    if isinstance(adapter_spec, dict):
        adapter = CommandBaseline(
            adapter_spec["command"],
            output / "baseline-index",
            adapter_spec.get("timeout", 1800),
            persistent=adapter_spec.get("persistent", False),
            server=server,
        )
    elif adapter_spec:
        adapter_factory = load_symbol(adapter_spec)
        adapter = (
            adapter_factory(server=server, working_dir=output / "baseline-index")
            if adapter_spec == "eval.graph_baseline:RAGAnything"
            else adapter_factory()
        )
    official_scorer = load_symbol(scorer_spec) if scorer_spec else None
    if graph_mode:
        if encoder is None:
            raise ValueError("HiVG-RAG variants require a multimodal encoder")
        builder = GraphBuilder(encoder, config.graph)
        retriever = Retriever(encoder, config.retrieval)
    rows, indices = [], []
    with (output / "predictions.jsonl").open("w") as stream:
        for raw in entry["docs"]:
            questions = [
                question
                for question in raw["questions"]
                if case_filter is None
                or question["id"] in case_filter
                or f"{raw['id']}/{question['id']}" in case_filter
            ]
            if not questions:
                continue
            checkpoint_key = _case_name(raw["id"], "")
            checkpoint = checkpoints.load(
                checkpoint_key, raw["id"], [question["id"] for question in questions]
            )
            if checkpoint is not None:
                indices.append(checkpoint["index"])
                for row in checkpoint["rows"]:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    rows.append(row)
                stream.flush()
                continue
            first_row = len(rows)
            index = None
            stamp = perf_counter()
            index_row: dict[str, Any] = {"doc_id": raw["id"]}
            failure = None
            try:
                if sensitivity_index is not None:
                    index = sensitivity_index(raw["id"], config.graph)
                    document = index.document
                else:
                    document = load_document(
                        raw, path.parent, cache, config.dpi, require_graph=graph_mode or stress
                    )
                    index_row["load_seconds"] = perf_counter() - stamp
                labels = raw.get("entity_gold", {})
                if stress:
                    document, labels = inject(
                        document,
                        labels,
                        raw["distractors"],
                        path.parent,
                        distractors,
                    )
                stamp = perf_counter()
                if graph_mode and index is None:
                    index = builder.build(document)
                elif adapter is not None:
                    external_index = adapter.index(document)
                if sensitivity_index is None:
                    index_row["index_seconds"] = perf_counter() - stamp
                if graph_mode:
                    index_row["after"] = topology(index, labels, config.graph.superhub_degree)
                    baseline = (
                        index
                        if before == config.graph
                        else sensitivity_index(raw["id"], before)
                        if sensitivity_index is not None
                        else GraphBuilder(encoder, before).build(
                            document,
                            features=(index.text, index.visual),
                        )
                    )
                    index_row["before"] = topology(baseline, labels, config.graph.superhub_degree)
            except Exception as error:  # noqa: BLE001 - record third-party failures per item
                failure = f"{type(error).__name__}: {error}"
                index_row["error"] = failure
            indices.append(index_row)
            for question in questions:
                row = {
                    **case,
                    "doc_id": raw["id"],
                    "question_id": question["id"],
                    "subset": question.get("subset", "unlabeled"),
                    "question": question["question"],
                    "answers": question["answers"],
                    "status": "ok",
                }
                try:
                    if failure:
                        raise RuntimeError(failure)
                    stamp = perf_counter()
                    query_image = None
                    query_image_path = (
                        path.parent / question["image"] if question.get("image") else None
                    )
                    if query_image_path is not None:
                        with Image.open(query_image_path) as source:
                            query_image = source.convert("RGB")
                    if graph_mode:
                        retrieved = retriever.retrieve(
                            index, question["question"], route, query_image=query_image
                        )
                    elif isinstance(adapter, CommandBaseline):
                        retrieved = adapter.retrieve(
                            external_index, question["question"], query_image=query_image_path
                        )
                    elif adapter is not None:
                        retrieved = adapter.retrieve(external_index, question["question"])
                    else:
                        pages = sorted(document.pages)
                        selected = _direct_pages(pages, direct_max_pages)
                        row.update(document_pages=len(pages), input_document_pages=len(selected))
                        retrieved = Retrieval(
                            [Evidence(f"page-{page}", f"page-{page}", page) for page in selected],
                            {"mode": "direct", "pages": selected},
                        )
                    retrieval_seconds = perf_counter() - stamp
                    trace_dir = (
                        output / "traces" / _case_name(raw["id"], question["id"]) if trace else None
                    )
                    generated = generator.generate(
                        document,
                        question["question"],
                        retrieved.evidence,
                        trace_dir,
                        seed=seed,
                        query_image=query_image,
                    )
                    elapsed = perf_counter() - stamp
                    row.update(
                        prediction=generated.content,
                        prompt_tokens=generated.prompt_tokens,
                        output_tokens=generated.output_tokens,
                        finish_reason=generated.finish_reason,
                        query_seconds=elapsed,
                        retrieval_seconds=retrieval_seconds,
                        generation_seconds=elapsed - retrieval_seconds,
                        **score(question, generated.content),
                    )
                    if "precomputed_retrieval_seconds" in retrieved.trace:
                        offline_retrieval = retrieved.trace["precomputed_retrieval_seconds"]
                        index_row["precomputed_index_seconds"] = retrieved.trace[
                            "precomputed_index_seconds"
                        ]
                        row["precomputed_retrieval_seconds"] = offline_retrieval
                        row["generation_plus_precomputed_retrieval_seconds"] = (
                            elapsed - retrieval_seconds + offline_retrieval
                        )
                    if official_scorer is not None:
                        official = float(official_scorer(question, generated.content))
                        if not 0 <= official <= 1:
                            raise ValueError(f"Official score out of range: {official}")
                        row["official_score"] = official
                    if trace_dir is not None:
                        if query_image is not None:
                            row["query_image_file"] = "query.png"
                        trace_dir.mkdir(parents=True, exist_ok=True)
                        row["trace_path"] = str(trace_dir.relative_to(output) / "trace.json")
                        _json(trace_dir / "trace.json", {**row, **_trace(index, retrieved)})
                except Exception as error:  # noqa: BLE001 - record third-party failures per item
                    row.update(status="error", error=f"{type(error).__name__}: {error}")
                    print(
                        f"{manifest['dataset']}/{method}/{raw['id']}/{question['id']}: "
                        f"{row['error']}",
                        file=sys.stderr,
                    )
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                rows.append(row)
            checkpoints.save(checkpoint_key, index_row, rows[first_row:])
    if adapter is not None and hasattr(adapter, "close"):
        adapter.close()
    _json(output / "documents.json", indices)
    _csv(
        output / "documents.csv",
        [
            {
                **{key: value for key, value in row.items() if key not in ("before", "after")},
                **{
                    f"{phase}_{key}": value
                    for phase in ("before", "after")
                    for key, value in row.get(phase, {}).items()
                },
            }
            for row in indices
        ],
    )
    _csv(output / "predictions.csv", rows)
    summary = summarize(rows, indices)
    _json(output / "summary.json", {**case, "dataset": manifest["dataset"], **summary})
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run HiVG-RAG experiments against a vLLM API")
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--servers", type=Path, default=Path("configs/servers.yaml"))
    parser.add_argument("--backbone", action="append")
    parser.add_argument(
        "--experiment",
        choices=(
            "main",
            "generalization",
            "ablation",
            "stress",
            "sensitivity",
            "efficiency",
            "case",
        ),
        default="main",
    )
    parser.add_argument("--method", choices=VARIANTS + BASELINES, action="append")
    parser.add_argument("--adapters", type=Path)
    parser.add_argument(
        "--scorer",
        action="append",
        help="module:function(question_dict, prediction) or DATASET=module:function",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resume", action="store_true", help="Reuse completed document checkpoints"
    )
    parser.add_argument("--render-cache", type=Path, default=Path("data/rendered"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stress-size", type=int, default=100)
    parser.add_argument(
        "--zoom-grid", type=float, nargs="+", default=[0.35, 0.45, 0.55, 0.65, 0.75]
    )
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--case-id", action="append", help="question ID or document ID/question ID")
    parser.add_argument(
        "--direct-max-pages", type=int, help="Cap Direct input pages with uniform position sampling"
    )
    options = parser.parse_args()
    if options.direct_max_pages is not None and options.direct_max_pages < 1:
        parser.error("--direct-max-pages must be positive")

    config = load_config(options.config)
    servers = yaml.safe_load(options.servers.read_text())
    adapters = {
        "visrag": "eval.baselines:VisRAG",
        "colpali": "eval.baselines:ColPali",
        "rag-anything": "eval.graph_baseline:RAGAnything",
        **((yaml.safe_load(options.adapters.read_text()) or {}) if options.adapters else {}),
    }
    defaults = {
        "main": ("hivg-rag", *BASELINES),
        "generalization": ("hivg-rag",),
        "ablation": VARIANTS,
        "stress": ("hivg-rag", "without-vger-split"),
        "sensitivity": ("hivg-rag",),
        "efficiency": ("hivg-rag", "direct"),
        "case": VARIANTS,
    }
    methods = options.method or defaults[options.experiment]
    for method in methods:
        if method in BASELINES and method != "direct" and method not in adapters:
            parser.error(
                f"Supply --adapters entry for {method}, or select available methods with --method"
            )
    backbones = options.backbone or (
        ["7b", "32b"] if options.experiment == "generalization" else ["7b"]
    )
    for backbone in backbones:
        if backbone not in servers or not SLUG.fullmatch(backbone):
            parser.error(f"Missing or invalid backbone {backbone}")
    if not all(0 <= value <= 1 for value in options.zoom_grid) or options.stress_size < 1:
        parser.error("Zoom thresholds must be in [0,1] and stress size must be positive")
    scorers = {}
    for value in options.scorer or ():
        name, found, symbol = value.partition("=")
        if found:
            scorers[name] = symbol
        else:
            scorers["*"] = name
    inputs = []
    stress = options.experiment == "stress"
    datasets = set()
    for path in options.manifest:
        manifest = _load_manifest(path)
        name = manifest["dataset"]
        if name in datasets:
            parser.error(f"Dataset slug {name} must be unique within a run")
        datasets.add(name)
        docs = manifest["documents"]
        if stress:
            docs = select_documents(docs, options.stress_size, options.seed)
        inputs.append({"path": path.resolve(), "data": manifest, "docs": docs})
    settings = [("default", config, 0)]
    if stress:
        settings = [(f"distractors-{n}", config, n) for n in LEVELS]
    if options.experiment == "sensitivity":
        settings = [
            (
                f"split-{split:g}_zoom-{zoom:g}",
                replace(
                    config,
                    graph=replace(config.graph, tau_split=split),
                    retrieval=replace(config.retrieval, tau_zoom=zoom),
                ),
                0,
            )
            for split, zoom in product(SPLIT_GRID, options.zoom_grid)
        ]
    options.output.mkdir(parents=True, exist_ok=options.resume)
    with lock_output(options.output):
        return _run_matrix(
            options,
            config,
            inputs,
            backbones,
            methods,
            settings,
            servers,
            adapters,
            scorers,
            stress,
        )


def _run_matrix(
    options, config, inputs, backbones, methods, settings, servers, adapters, scorers, stress
):
    encoder = (
        SiglipEncoder(config.encoder) if any(method in VARIANTS for method in methods) else None
    )
    summaries = []
    active_manifest = None
    sensitivity_index = None
    for entry, backbone, method, (tag, config_at_setting, level) in product(
        inputs,
        backbones,
        methods,
        settings,
    ):
        dataset = entry["data"]["dataset"]
        if (
            options.experiment == "sensitivity"
            and encoder is not None
            and dataset != active_manifest
        ):
            sensitivity_index = _sensitivity_indices(entry, config, options.render_cache, encoder)
            active_manifest = dataset
        print(f"{dataset}/{backbone}/{method}/{tag}", file=sys.stderr, flush=True)
        summary = run(
            entry,
            _changes(config_at_setting, method),
            method,
            tag,
            backbone,
            servers[backbone],
            options.output / dataset / backbone / method / tag,
            options.render_cache,
            encoder,
            trace=options.trace or options.experiment == "case",
            distractors=level,
            stress=stress,
            seed=options.seed,
            adapter_spec=adapters.get(method),
            scorer_spec=scorers.get(dataset, scorers.get("*")),
            case_filter=set(options.case_id) if options.case_id else None,
            direct_max_pages=options.direct_max_pages,
            sensitivity_index=sensitivity_index if method in VARIANTS else None,
            resume=options.resume,
        )
        for subset in ("overall", "macro", "micro", "unlabeled"):
            summaries.append(
                {
                    "dataset": dataset,
                    "backbone": backbone,
                    "method": method,
                    "setting": tag,
                    "distractors": level if stress else None,
                    "subset": subset,
                    "tau_split": config_at_setting.graph.tau_split,
                    "tau_zoom": config_at_setting.retrieval.tau_zoom,
                    **summary[subset],
                    "index_seconds": summary["indexing"]["index_seconds"],
                    "load_seconds": summary["indexing"]["load_seconds"],
                    **{
                        key: value
                        for key, value in summary["indexing"].items()
                        if key == "precomputed_index_seconds"
                    },
                    **{
                        f"{phase}_{metric}": summary.get(f"topology_{phase}", {}).get(metric)
                        for phase in ("before", "after")
                        for metric in ("superhubs", "entity_precision")
                    },
                }
            )
        _json(options.output / "summary.json", summaries)
        _csv(options.output / "summary.csv", summaries)
    overall = [row for row in summaries if row["subset"] == "overall"]
    return int(not overall or any(row["errors"] or not row["scored"] for row in overall))


if __name__ == "__main__":
    raise SystemExit(main())
