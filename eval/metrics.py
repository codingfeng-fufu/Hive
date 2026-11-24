from collections.abc import Iterable
from statistics import mean

from src.graph_builder import GraphIndex, predicted_links


def _normalized(value: str) -> str:
    return " ".join(str(value).casefold().split())


def anls(prediction: str, reference: str) -> float:
    if not prediction or not reference:
        return float(prediction == reference)
    previous = list(range(len(reference) + 1))
    for i, left in enumerate(prediction, 1):
        current = [i]
        for j, right in enumerate(reference, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (left != right))
            )
        previous = current
    distance = previous[-1] / max(len(prediction), len(reference))
    return 1 - distance if distance < 0.5 else 0.0


def score(question: dict, prediction: str) -> dict[str, float]:
    reference = [_normalized(answer) for answer in question["answers"]]
    normalized = _normalized(prediction)
    return {
        "accuracy": float(normalized in reference),
        "anls": max(anls(normalized, answer) for answer in reference),
    }


def topology(index: GraphIndex, labels: dict[str, str], threshold: int) -> dict:
    eligible = {node.id for node in index.document.mentions if node.kind == "entity"}
    if missing := labels.keys() - eligible:
        raise ValueError(f"Gold entity IDs absent from graph: {sorted(missing)}")
    merged, aliases = predicted_links(index, set(labels))
    degree = {node: index.micro.degree[node] for node in {index.entity_of[key] for key in eligible}}
    result = {
        "macro_nodes": len(index.macro),
        "micro_nodes": len(index.micro),
        "micro_edges": index.micro.number_of_edges(),
        "split_pairs": len(index.splits),
        "alias_edges": sum(
            "VISUAL_ALIAS" in edge["weights"] for edge in index.micro.edges.values()
        ),
        "superhubs": sum(value >= threshold for value in degree.values()),
        "max_entity_degree": max(degree.values(), default=0),
        "annotated_mentions": len(labels),
    }
    for name, pairs in [("merge", merged), ("alias", aliases), ("entity", merged | aliases)]:
        correct = sum(labels[left] == labels[right] for left, right in pairs)
        result[f"{name}_correct"] = correct
        result[f"{name}_predicted"] = len(pairs)
        result[f"{name}_precision"] = correct / len(pairs) if pairs else None
    return result


def _average(values: Iterable[float | None]) -> float | None:
    observed = [value for value in values if value is not None]
    return mean(observed) if observed else None


def summarize(rows: list[dict], indices: list[dict]) -> dict:
    summary = {}
    for subset in ("overall", "macro", "micro", "unlabeled"):
        group = [row for row in rows if subset == "overall" or subset == row["subset"]]
        valid = [row for row in group if row["status"] == "ok"]
        summary[subset] = {
            "n": len(group),
            "scored": len(valid),
            "errors": len(group) - len(valid),
            "prompt_token_count_n": sum(row.get("prompt_tokens") is not None for row in valid),
            "truncated": sum(row.get("finish_reason") == "length" for row in valid),
            **{
                key: _average(row.get(key) for row in valid)
                for key in (
                    "accuracy",
                    "anls",
                    "official_score",
                    "query_seconds",
                    "retrieval_seconds",
                    "generation_seconds",
                    "prompt_tokens",
                    "output_tokens",
                )
            },
        }
        direct = [row for row in valid if "document_pages" in row]
        if direct:
            summary[subset]["page_capped"] = sum(
                row["input_document_pages"] < row["document_pages"] for row in direct
            )
            summary[subset]["mean_input_document_pages"] = mean(
                row["input_document_pages"] for row in direct
            )
        for key in (
            "precomputed_retrieval_seconds",
            "generation_plus_precomputed_retrieval_seconds",
        ):
            if (value := _average(row.get(key) for row in valid)) is not None:
                summary[subset][key] = value
    summary["indexing"] = {
        "documents": len(indices),
        "index_seconds": _average(index.get("index_seconds") for index in indices),
        "load_seconds": _average(index.get("load_seconds") for index in indices),
    }
    if (value := _average(index.get("precomputed_index_seconds") for index in indices)) is not None:
        summary["indexing"]["precomputed_index_seconds"] = value
    for phase in ("before", "after"):
        graphs = [index[phase] for index in indices if index.get(phase) is not None]
        if not graphs:
            continue
        result = {"documents": len(graphs)}
        for key in (
            "superhubs",
            "max_entity_degree",
            "micro_nodes",
            "micro_edges",
            "alias_edges",
            "split_pairs",
            "annotated_mentions",
        ):
            result[key] = _average(graph[key] for graph in graphs)
        for key in ("entity", "merge", "alias"):
            correct = sum(graph[f"{key}_correct"] for graph in graphs)
            pairs = sum(graph[f"{key}_predicted"] for graph in graphs)
            result.update(
                {
                    f"{key}_precision": correct / pairs if pairs else None,
                    f"{key}_predicted": pairs,
                    f"{key}_correct": correct,
                }
            )
        summary[f"topology_{phase}"] = result
    return summary
