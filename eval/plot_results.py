import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(rows: list[dict], figure: str, metric: str, output: Path) -> None:
    if not rows:
        raise ValueError("No complete result groups match the requested figure")
    groups = defaultdict(list)
    for row in rows:
        groups[row["dataset"], row["backbone"]].append(row)
    fig, axes = plt.subplots(1, len(groups), figsize=(4.8 * len(groups), 3.65), squeeze=False)
    for axis, ((dataset, backbone), group) in zip(axes[0], sorted(groups.items())):
        axis.set_title(f"{dataset} · {backbone}")
        if figure == "stress":
            by_method = defaultdict(list)
            for row in group:
                by_method[row["method"]].append((row["distractors"], row[metric]))
            for index, (method, observations) in enumerate(sorted(by_method.items())):
                axis.plot(
                    *zip(*sorted(observations)),
                    marker=("o", "x", "s", "^")[index % 4],
                    linestyle=("-", "--", "-.", ":")[index % 4],
                    label=method,
                )
            axis.set_xticks([0, 1, 2, 4])
            axis.set_xlabel("Distractor pages")
            axis.set_ylabel(metric.replace("_", " ").upper())
            axis.set_ylim(0, 1.02)
            axis.legend(frameon=False, fontsize="small")
            axis.grid(axis="y", alpha=0.2)
        elif figure == "pareto":
            points = defaultdict(list)
            for row in group:
                points[row["prompt_tokens"], row[metric]].append(row["method"])
            frontier = []
            best = -1.0
            for tokens, value in sorted(points, key=lambda item: (item[0], -item[1])):
                axis.scatter(tokens, value)
                axis.annotate(
                    ", ".join(sorted(points[tokens, value])),
                    (tokens, value),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize="x-small",
                )
                if value > best:
                    frontier.append((tokens, value))
                    best = value
            if frontier:
                axis.plot(*zip(*frontier), color="0.6", linewidth=1, zorder=0)
            axis.set_xlabel("Mean multimodal prompt tokens")
            axis.set_ylabel(metric.replace("_", " ").upper())
            axis.grid(axis="y", alpha=0.2)
        else:
            split = sorted({row["tau_split"] for row in group})
            zoom = sorted({row["tau_zoom"] for row in group})
            grid = np.full((len(split), len(zoom)), np.nan)
            for row in group:
                grid[split.index(row["tau_split"]), zoom.index(row["tau_zoom"])] = row[metric]
            image = axis.imshow(grid, aspect="auto", vmin=0, vmax=1, cmap="viridis")
            for (i, j), value in np.ndenumerate(grid):
                if np.isfinite(value):
                    axis.text(
                        j,
                        i,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color="white" if value < 0.48 else "black",
                        fontsize="small",
                    )
            axis.set_xticks(range(len(zoom)), [f"{value:g}" for value in zoom])
            axis.set_yticks(range(len(split)), [f"{value:g}" for value in split])
            axis.set(xlabel=r"$\tau_{zoom}$", ylabel=r"$\tau_{split}$")
            fig.colorbar(image, ax=axis, label=metric.replace("_", " ").upper())
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot matched HiVG-RAG benchmark results")
    parser.add_argument("summary", type=Path)
    parser.add_argument("figure", choices=("stress", "pareto", "sensitivity"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--subset", choices=("overall", "macro", "micro"), default="overall")
    parser.add_argument(
        "--metric", choices=("accuracy", "anls", "official_score"), default="accuracy"
    )
    parser.add_argument("--dataset")
    parser.add_argument("--backbone")
    options = parser.parse_args()
    rows = [
        row
        for row in json.loads(options.summary.read_text())
        if row["subset"] == options.subset
        and row["scored"]
        and not row["errors"]
        and row.get(options.metric) is not None
        and (not options.dataset or row["dataset"] == options.dataset)
        and (not options.backbone or row["backbone"] == options.backbone)
    ]
    if options.figure == "stress":
        rows = [row for row in rows if row["distractors"] is not None]
    elif options.figure == "sensitivity":
        rows = [
            row
            for row in rows
            if row["method"] == "hivg-rag" and row["setting"].startswith("split-")
        ]
    else:
        rows = [
            row
            for row in rows
            if row["setting"] == "default" and row["prompt_token_count_n"] == row["scored"]
        ]
    plot(rows, options.figure, options.metric, options.output)


if __name__ == "__main__":
    main()
