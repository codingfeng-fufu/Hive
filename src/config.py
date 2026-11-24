from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class EncoderConfig:
    model: str = "google/siglip-so400m-patch14-384"
    revision: str | None = None
    device: str = "auto"
    batch_size: int = 8


@dataclass(frozen=True)
class GraphConfig:
    tau_text: float = 0.85
    tau_split: float = 0.60
    tau_merge: float = 0.85
    alias_page_window: int = 1
    alias_context_min: float = 0.55
    macro_edge_min: float = 0.45
    macro_neighbors: int = 4
    micro_neighbors: int = 4
    superhub_degree: int = 20
    enable_split: bool = True
    enable_bridge: bool = True


@dataclass(frozen=True)
class RetrievalConfig:
    lambda_lex: float = 0.65
    tau_zoom: float = 0.55
    temperature: float = 0.10
    visual_weight: float = 0.40
    query_image_weight: float = 0.50
    top_parents: int = 2
    macro_k: int = 6
    micro_k: int = 8
    ppr_alpha: float = 0.85
    native_crop: bool = True


@dataclass(frozen=True)
class Config:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    dpi: int = 250


def load_config(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text()) or {}
    config = Config(
        encoder=EncoderConfig(**raw.pop("encoder", {})),
        graph=GraphConfig(**raw.pop("graph", {})),
        retrieval=RetrievalConfig(**raw.pop("retrieval", {})),
        **raw,
    )
    graph, route = config.graph, config.retrieval
    if any(
        not 0 <= value <= 1
        for value in (
            graph.tau_text,
            graph.tau_split,
            graph.tau_merge,
            graph.alias_context_min,
            graph.macro_edge_min,
            route.lambda_lex,
            route.tau_zoom,
            route.visual_weight,
            route.query_image_weight,
        )
    ):
        raise ValueError("Cosine thresholds, zoom threshold, and mixing weights must be in [0, 1]")
    if not (0 < route.ppr_alpha < 1 and route.temperature > 0):
        raise ValueError("PPR alpha must be in (0, 1) and temperature must be positive")
    if (
        min(
            config.dpi,
            config.encoder.batch_size,
            route.top_parents,
            route.macro_k,
            route.micro_k,
            graph.superhub_degree,
        )
        < 1
        or min(graph.alias_page_window, graph.macro_neighbors, graph.micro_neighbors) < 0
    ):
        raise ValueError("Counts, batch size, and render DPI must be in range")
    return config
