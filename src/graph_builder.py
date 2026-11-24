from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

import networkx as nx
import numpy as np

from .config import GraphConfig
from .document import Document
from .embeddings import Encoder, Vector, normalize


@dataclass(frozen=True)
class Pair:
    left: str
    right: str
    text: float
    visual: float | None


@dataclass(frozen=True)
class Link:
    left: str
    right: str
    kind: str
    weight: float


def add_link(graph: nx.Graph, left: str, right: str, kind: str, weight: float) -> None:
    if left == right:
        return
    if graph.has_edge(left, right):
        edge = graph[left][right]
        edge["weights"][kind] = max(edge["weights"].get(kind, 0), weight)
        edge["weight"] = max(edge["weights"].values())
    else:
        graph.add_edge(left, right, weight=weight, weights={kind: weight})


@dataclass
class GraphIndex:
    document: Document
    macro: nx.Graph
    micro: nx.Graph
    hierarchy: nx.DiGraph
    text: dict[str, Vector]
    visual: dict[str, Vector]
    entity_of: dict[str, str]
    splits: tuple[Pair, ...]
    aliases: tuple[Link, ...]
    links: tuple[Link, ...]
    mentions_by_parent: dict[str, tuple[str, ...]]
    links_by_parent: dict[str, tuple[Link, ...]]

    def project_micro(self, parents: set[str] | None = None) -> nx.Graph:
        if parents is None:
            return self.micro
        active = {
            mention
            for parent in parents
            for mention in self.mentions_by_parent.get(parent, ())
            if self.hierarchy.has_edge(self.entity_of[mention], parent)
        }
        candidates = {link for parent in parents for link in self.links_by_parent.get(parent, ())}
        return _micro_graph(self.entity_of, active, candidates)


def _micro_graph(
    entity_of: dict[str, str],
    active: set[str],
    links: set[Link] | tuple[Link, ...],
) -> nx.Graph:
    members = defaultdict(list)
    for mention in sorted(active):
        members[entity_of[mention]].append(mention)
    graph = nx.Graph()
    graph.add_nodes_from((node, {"members": tuple(ids)}) for node, ids in sorted(members.items()))
    for link in sorted(links, key=lambda item: (item.left, item.right, item.kind)):
        if link.left in active and link.right in active:
            add_link(graph, entity_of[link.left], entity_of[link.right], link.kind, link.weight)
    return graph


class GraphBuilder:
    def __init__(self, encoder: Encoder, config: GraphConfig) -> None:
        self.encoder = encoder
        self.config = config

    def build(
        self,
        doc: Document,
        *,
        features: tuple[dict[str, Vector], dict[str, Vector]] | None = None,
    ) -> GraphIndex:
        macros = {node.id: node for node in doc.macro}
        mentions = {node.id: node for node in doc.mentions}
        nodes = {**macros, **mentions}
        if len(nodes) != len(doc.macro) + len(doc.mentions):
            raise ValueError(f"{doc.id}: node IDs must be unique across layers")
        if not macros or any(node.parent not in macros for node in mentions.values()):
            raise ValueError(f"{doc.id}: expected macro nodes and a valid parent for every mention")
        ids = sorted(nodes)
        if features is None:
            text = dict(zip(ids, normalize(self.encoder.text([nodes[key].text for key in ids]))))
            visual = self._visual_features(doc)
        else:
            text, visual = features

        macro = nx.Graph()
        macro.add_nodes_from(sorted(macros))
        for node in macros.values():
            for target in node.references:
                if target not in macros:
                    raise ValueError(f"{doc.id}: unknown reference {target}")
                add_link(macro, node.id, target, "REFERENCE", 1)
        links = _nearest(
            sorted(macros), text, self.config.macro_neighbors, self.config.macro_edge_min
        )
        for left, right, weight in links:
            add_link(macro, left, right, "SEMANTIC", weight)

        entity_of, splits, aliases = self._resolve(doc, macro, text, visual)
        groups = defaultdict(list)
        hierarchy = nx.DiGraph()
        hierarchy.add_nodes_from(sorted(macros), layer="macro")
        hierarchy.add_nodes_from(sorted(set(entity_of.values())), layer="micro")
        for mention in mentions.values():
            groups[mention.parent].append(mention.id)
            entity = entity_of[mention.id]
            if not hierarchy.has_edge(entity, mention.parent):
                hierarchy.add_edge(entity, mention.parent, kind="PART_OF", mentions=[])
            hierarchy[entity][mention.parent]["mentions"].append(mention.id)
        links = (
            tuple(
                Link(left, right, "LOCAL", weight)
                for group in groups.values()
                for left, right, weight in _nearest(
                    sorted(group), text, self.config.micro_neighbors, 0.0
                )
            )
            + aliases
        )
        links_by_parent = defaultdict(list)
        for link in links:
            for parent in {mentions[link.left].parent, mentions[link.right].parent}:
                links_by_parent[parent].append(link)
        return GraphIndex(
            doc,
            macro,
            _micro_graph(entity_of, set(mentions), links),
            hierarchy,
            text,
            visual,
            entity_of,
            splits,
            aliases,
            links,
            {parent: tuple(ids) for parent, ids in groups.items()},
            {parent: tuple(group) for parent, group in links_by_parent.items()},
        )

    def _visual_features(self, doc: Document) -> dict[str, Vector]:
        nodes = [node for node in (*doc.macro, *doc.mentions) if node.region is not None]
        features = {}
        for start in range(0, len(nodes), 16):
            batch = nodes[start : start + 16]
            regions = [node.region for node in batch if node.region is not None]
            features.update(
                zip(
                    (node.id for node in batch),
                    normalize(self.encoder.image([doc.crop(region) for region in regions])),
                )
            )
        return features

    def _resolve(
        self,
        doc: Document,
        macro: nx.Graph,
        text: dict[str, Vector],
        visual: dict[str, Vector],
    ) -> tuple[dict[str, str], tuple[Pair, ...], tuple[Link, ...]]:
        config = self.config
        entities = {node.id: node for node in doc.mentions if node.kind == "entity"}
        macros = {node.id: node for node in doc.macro}
        clusters = {node.id: {node.id} for node in doc.mentions}
        entity_of = {key: key for key in clusters}
        candidates, splits, bridges = [], [], []
        ids = sorted(entities)
        matrix = np.stack([text[key] for key in ids]) if ids else np.empty((0, 0))
        visual_matrix = np.zeros_like(matrix)
        grounded = np.array([key in visual for key in ids], dtype=bool)
        for i in np.flatnonzero(grounded):
            visual_matrix[i] = visual[ids[i]]
        for i, left in enumerate(ids):
            if i == len(ids) - 1:
                break
            sims = np.clip(matrix[i + 1 :] @ matrix[i], -1, 1)
            if grounded[i]:
                vis = np.clip(visual_matrix[i + 1 :] @ visual_matrix[i], -1, 1)
                possible_alias = grounded[i + 1 :] & (vis >= config.tau_merge)
            else:
                vis = np.zeros_like(sims)
                possible_alias = np.zeros(len(sims), dtype=bool)
            if not config.enable_bridge:
                possible_alias[:] = False
            for offset in np.flatnonzero((sims >= config.tau_text) | possible_alias):
                j = i + 1 + int(offset)
                right = ids[j]
                pair = Pair(
                    left,
                    right,
                    float(sims[offset]),
                    float(vis[offset]) if grounded[i] and grounded[j] else None,
                )
                if pair.text >= config.tau_text:
                    if (
                        config.enable_split
                        and pair.visual is not None
                        and pair.visual < config.tau_split
                    ):
                        splits.append(pair)
                    else:
                        candidates.append(pair)
                elif (
                    pair.visual is not None
                    and pair.visual >= config.tau_merge
                    and _related(
                        macros[entities[left].parent],
                        macros[entities[right].parent],
                        macro,
                        text,
                        config,
                    )
                ):
                    bridges.append(Link(left, right, "VISUAL_ALIAS", pair.visual))

        def compatible(left: str, right: str) -> bool:
            if np.clip(text[left] @ text[right], -1, 1) < config.tau_text:
                return False
            if not config.enable_split or left not in visual or right not in visual:
                return True
            return bool(np.clip(visual[left] @ visual[right], -1, 1) >= config.tau_split)

        def priority(pair: Pair) -> tuple:
            first, second = entities[pair.left], entities[pair.right]
            return (
                pair.visual is None,
                -pair.text,
                -pair.visual if pair.visual is not None else 0,
                first.parent != second.parent,
                abs(first.page - second.page),
                pair.left,
                pair.right,
            )

        for pair in sorted(candidates, key=priority):
            left, right = sorted((entity_of[pair.left], entity_of[pair.right]))
            if left == right:
                continue
            if not all(compatible(a, b) for a in clusters[left] for b in clusters[right]):
                continue
            for member in clusters.pop(right):
                entity_of[member] = left
                clusters[left].add(member)
        return entity_of, tuple(splits), tuple(bridges)


def _related(left, right, macro: nx.Graph, text: dict[str, Vector], config: GraphConfig) -> bool:
    if left.id == right.id or left.page == right.page:
        return True
    if "REFERENCE" in macro.get_edge_data(left.id, right.id, {}).get("weights", {}):
        return True
    if abs(left.page - right.page) > config.alias_page_window:
        return False
    return bool(
        (left.section and left.section == right.section)
        or text[left.id] @ text[right.id] >= config.alias_context_min
    )


def _nearest(ids: list[str], text: dict[str, Vector], k: int, minimum: float):
    if len(ids) < 2 or not k:
        return
    matrix = np.stack([text[key] for key in ids])
    for i, left in enumerate(ids):
        scores = np.clip(matrix @ matrix[i], -1, 1)
        scores[i] = -np.inf
        for j in np.argsort(-scores, kind="stable")[:k]:
            if scores[j] >= minimum and scores[j] > 0:
                yield left, ids[j], float(scores[j])


def predicted_links(
    index: GraphIndex, ids: set[str]
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    merged = {
        tuple(sorted(pair))
        for node in index.micro.nodes.values()
        for pair in combinations(ids.intersection(node["members"]), 2)
    }
    aliases = {
        tuple(sorted((link.left, link.right)))
        for link in index.aliases
        if link.left in ids and link.right in ids
    }
    return merged, aliases
