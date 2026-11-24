"""Plot complete pilot QA runs with measured multimodal input token counts."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, PercentFormatter

METHODS = {
    "hivg-rag": ("HiVG-RAG", "D", "#00796F"),
    "direct": ("Direct", "o", "#374151"),
    "visrag": ("VisRAG", "v", "#A23B72"),
    "colpali": ("ColPali", "s", "#6740A0"),
    "macro-only": ("Macro-only", "^", "#B55617"),
    "micro-only": ("Micro-only", "P", "#1D61A3"),
    "without-native-crop": ("No native crop", "X", "#6B7280"),
    "without-vger-split": ("No VGER-Split", "+", "#6B7280"),
    "without-vger-bridge": ("No VGER-Bridge", "x", "#6B7280"),
    "rag-anything": ("RAG-Anything", "h", "#988317"),
}
DATASETS = {
    "mmlongbench-doc-full-document-pilot": ("MMLongBench-Doc", 68),
    "docbench-full-document-pilot": ("DocBench", 20),
}


def plot(rows: list[dict], output: Path, metric: str = "anls") -> None:
    plt.rcParams.update({"font.family": "STIXGeneral", "font.size": 11, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.1), sharex=True, sharey=True)
    largest_context = max(row["subsets"]["overall"]["prompt_tokens"] or 0 for row in rows)
    ticks = [300, 1200, 4800, 19200] if largest_context > 5500 else [300, 600, 1200, 2400, 4800]
    shown = set()
    for row_axes, (dataset, (title, count)) in zip(axes, DATASETS.items()):
        for axis, model in zip(row_axes, ("7b", "32b")):
            points = []
            for row in rows:
                if (row["dataset"], row["model"], row["setting"]) != (dataset, model, "default"):
                    continue
                method = row["method"]
                stats = row["subsets"]["overall"]
                if stats["scored"] != count or stats["prompt_token_count_n"] != count:
                    continue
                _, marker, color = METHODS[method]
                tokens, score = stats["prompt_tokens"], stats[metric]
                axis.scatter(tokens, score, marker=marker, color=color, s=37, zorder=3)
                points.append((tokens, score))
                shown.add(method)
            if not points:
                raise ValueError(f"No complete matched runs for {dataset}/{model}")
            frontier = []
            for tokens, score in sorted(points, key=lambda point: (point[0], -point[1])):
                if not frontier or score > frontier[-1][1]:
                    frontier.append((tokens, score))
            axis.plot(*zip(*frontier), linewidth=1, color="0.63", zorder=1)
            axis.set_title(f"{title} · {model.upper()} ($n={count}$)")
            axis.set(xscale="log")
            axis.xaxis.set_major_locator(FixedLocator(ticks))
            axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
            axis.yaxis.set_major_locator(MaxNLocator(5))
            if metric == "accuracy":
                axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            axis.tick_params(which="minor", bottom=False)
            axis.grid(axis="y", color="0.9")
            axis.spines[["top", "right"]].set_visible(False)
        row_axes[0].set_ylabel("ANLS" if metric == "anls" else "Exact-match accuracy")
    fig.supxlabel("Mean multimodal prompt tokens (log scale)", y=0.165)
    handles = [
        Line2D([], [], marker=marker, color=color, linestyle="none", label=label)
        for method, (label, marker, color) in METHODS.items()
        if method in shown
    ]
    handles.append(Line2D([], [], color="0.63", label="Empirical Pareto frontier"))
    fig.legend(handles=handles, loc="lower center", ncols=4, frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.095, right=0.965, top=0.93, bottom=0.26, hspace=0.43, wspace=0.15)
    fig.savefig(output)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--metric", choices=("anls", "accuracy"), default="anls")
    args = parser.parse_args()
    plot(json.loads(args.results.read_text()), args.output, args.metric)
