import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import networkx as nx
import numpy as np
from PIL import Image

from .config import RetrievalConfig
from .document import Region
from .embeddings import Encoder, Vector, normalize
from .graph_builder import GraphIndex

Route = Literal["adaptive", "macro", "micro"]

_DETAIL = re.compile(
    r"\b(?:dashed|dotted|legend|axis|axes|tick|cell|bottom[- ]right|top[- ]left|"
    r"top[- ]right|bottom[- ]left|inset|at the intersection|how many|"
    r"(?:which|what) (?:row|column|color|colour|value|percentage)|"
    r"(?:row|column|panel|fig(?:ure)?|table|slide|page|part)\.?\s*[A-Z]?\d+|"
    r"\d+(?:[.,]\d+)?\s*(?:%|mm|cm|kg|Hz|kW))(?!\w)",
    re.IGNORECASE,
)
_MACRO = re.compile(
    r"\b(?:overall|summari[sz]e|overview|main (?:theme|purpose)|across (?:the|all)|"
    r"(?:in|among|of) all (?:\d+ )?(?:pages|chapters|references|charts|sections|slides|figures|tables))\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Evidence:
    node_id: str
    occurrence_id: str
    page: int | None
    text: str = ""
    region: Region | None = None
    score: float | None = None
    include_image: bool = True
    image_path: Path | None = None


@dataclass
class Retrieval:
    evidence: list[Evidence]
    trace: dict


def _probabilities(scores: dict[str, float], temperature: float) -> dict[str, float]:
    peak = max(scores.values())
    weights = {node: math.exp((score - peak) / temperature) for node, score in scores.items()}
    total = sum(weights.values())
    return {node: weight / total for node, weight in weights.items()}


def _rank(scores: dict[str, float]) -> list[str]:
    return sorted(scores, key=lambda node: (-scores[node], node))


def _ppr(graph: nx.Graph, scores: dict[str, float], config: RetrievalConfig) -> dict[str, float]:
    if not scores:
        return {}
    weights = _probabilities(scores, config.temperature)
    return nx.pagerank(
        graph,
        alpha=config.ppr_alpha,
        personalization=weights,
        dangling=weights,
        max_iter=250,
        tol=1e-10,
    )


def lexical_prior(question: str) -> float:
    if _MACRO.search(question):
        return 0.30 if _DETAIL.search(question) else 0.0
    if _DETAIL.search(question):
        return 1.0
    if re.search(r"\d", question):
        return 0.50
    return 0.0


class Retriever:
    def __init__(self, encoder: Encoder, config: RetrievalConfig) -> None:
        self.encoder = encoder
        self.config = config

    def _similarities(
        self,
        index: GraphIndex,
        nodes: list[str],
        query: Vector,
    ) -> dict[str, float]:
        if not nodes:
            return {}
        scores = np.clip(np.asarray([index.text[node] for node in nodes]) @ query, -1, 1)
        grounded = [i for i, node in enumerate(nodes) if node in index.visual]
        if grounded:
            visual = np.clip(np.asarray([index.visual[nodes[i]] for i in grounded]) @ query, -1, 1)
            visual_std = max(float(visual.std()), 1e-6)
            # Text-to-text and text-to-image cosine scores have different scales.
            aligned = np.clip(
                scores.mean() + (visual - visual.mean()) * scores.std() / visual_std,
                -1,
                1,
            )
            scores[grounded] = (1 - self.config.visual_weight) * scores[
                grounded
            ] + self.config.visual_weight * np.maximum(scores[grounded], aligned)
        return {node: float(score) for node, score in zip(nodes, scores)}

    def retrieve(
        self,
        index: GraphIndex,
        question: str,
        route: Route = "adaptive",
        *,
        query_image: Image.Image | None = None,
    ) -> Retrieval:
        config = self.config
        query = normalize(self.encoder.text([question]))[0]
        if query_image is not None:
            visual = normalize(self.encoder.image([query_image]))[0]
            weight = config.query_image_weight
            query = normalize((1 - weight) * query + weight * visual)
        macro_scores = self._similarities(index, list(index.macro), query)
        probabilities = _probabilities(macro_scores, config.temperature)
        entropy = -sum(p * math.log(p) for p in probabilities.values() if p > 0)
        focus = 1 - entropy / math.log(len(probabilities)) if len(probabilities) > 1 else 0.0
        prior = lexical_prior(question)
        gamma = float(np.clip(config.lambda_lex * prior + (1 - config.lambda_lex) * focus, 0, 1))
        detail = route == "micro" or (route == "adaptive" and gamma >= config.tau_zoom)
        parents = _rank(macro_scores)[: config.top_parents] if detail and route != "micro" else []
        trace = {
            "gamma": gamma,
            "lexical_prior": prior,
            "entropy": entropy,
            "mode": "micro" if detail else "macro",
            "ranked_parents": _rank(macro_scores),
            "selected_parents": parents,
            "macro_probabilities": probabilities,
        }
        if not detail:
            macros = {node.id: node for node in index.document.macro}
            ranking = _ppr(index.macro, macro_scores, config)
            selected = _rank(ranking)[: config.macro_k]
            trace.update(selected_parents=selected, visited_nodes=len(index.macro))
            return Retrieval(
                [
                    Evidence(
                        node,
                        node,
                        macros[node].page,
                        macros[node].text,
                        macros[node].region,
                        ranking[node],
                    )
                    for node in selected
                ],
                trace,
            )

        graph = index.project_micro(set(parents) if route != "micro" else None)
        mentions = {node.id: node for node in index.document.mentions}
        relevance = self._similarities(
            index,
            [member for data in graph.nodes.values() for member in data["members"]],
            query,
        )
        best = {
            node: _rank({member: relevance[member] for member in data["members"]})[0]
            for node, data in graph.nodes(data=True)
        }
        ranking = _ppr(graph, {node: relevance[member] for node, member in best.items()}, config)
        grounded_by_node = {
            node: _rank(grounded)[0]
            for node, data in graph.nodes(data=True)
            if (
                grounded := {
                    member: relevance[member]
                    for member in data["members"]
                    if mentions[member].region is not None
                }
            )
        }
        ranked = _rank(ranking)
        selected = ranked[: config.micro_k]
        if selected and grounded_by_node and not any(node in grounded_by_node for node in selected):
            selected[-1] = next(node for node in ranked if node in grounded_by_node)
        evidence = []
        images = set()
        for node in selected:
            occurrence = grounded_by_node.get(node, best[node])
            mention = mentions[occurrence]
            include_image = (
                config.native_crop and mention.region is not None and mention.region not in images
            )
            evidence.append(
                Evidence(
                    node,
                    occurrence,
                    mention.page,
                    mention.text,
                    mention.region,
                    ranking[node],
                    include_image,
                )
            )
            if include_image and mention.region is not None:
                images.add(mention.region)
        trace.update(visited_nodes=len(graph), visited_edges=graph.number_of_edges())
        return Retrieval(evidence, trace)
