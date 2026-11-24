"""Draw held-out VGER pairwise cases from the original PDF crops and scored CSV."""

import argparse
import ast
import csv
from pathlib import Path

import fitz
import matplotlib

from eval.vger_pilot import _crop

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CASES = (
    ("Split repeated panel label", "2405.09818v1", "same_name_different", "(a)", "(a)", 6, 10),
    (
        "Recover alias across pages",
        "camry_ebrochure",
        "different_name_same",
        "SE Hybrid Nightshade",
        "Camry SE Nightshade",
        2,
        5,
    ),
    (
        "False alias between crops",
        "camry_ebrochure",
        "different_name_different",
        "SE Hybrid Nightshade",
        "Camry",
        2,
        5,
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs", type=Path)
    parser.add_argument("pdfs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.pairs.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    plt.rcParams.update({"font.family": "STIXGeneral", "font.size": 10, "pdf.fonttype": 42})
    figure, axes = plt.subplots(3, 3, figsize=(7.2, 5.6), gridspec_kw={"width_ratios": [1, 1, 1.2]})
    for panel, (title, doc, stratum, left, right, lp, rp) in zip(axes, CASES):
        matched = [
            row
            for row in rows
            if row["split"] == "heldout"
            and row["document"] == doc
            and row["stratum"] == stratum
            and (row["left_label"], row["right_label"]) == (left, right)
            and (int(row["left_page"]), int(row["right_page"])) == (lp, rp)
        ]
        if title.startswith("Recover"):
            matched = [row for row in matched if row["defaults_figure"] == "none"]
        (row,) = matched
        with fitz.open(args.pdfs / f"{doc}.pdf") as pdf:
            for axis, side in zip(panel, ("left", "right")):
                box = tuple(ast.literal_eval(row[f"{side}_bbox"]))
                page = int(row[f"{side}_page"])
                axis.imshow(_crop(pdf, page, box))
                axis.set_title(f"{row[f'{side}_label']} · p. {page}", fontsize=10, pad=5)
                axis.axis("off")
        summary = panel[2]
        summary.axis("off")
        summary.text(0, 0.97, title, va="top", weight="bold", fontsize=11)
        for y, (score, label) in zip(
            (0.72, 0.52, 0.32), [("text", "Text"), ("figure", "Full figure"), ("crop", "Crop")]
        ):
            summary.text(
                0, y, f"{label:<13} {float(row[score]):.3f}  ·  {row[f'defaults_{score}']}"
            )
        summary.text(0, 0.08, doc + ".pdf", color="0.35", fontsize=9)
    figure.subplots_adjust(
        left=0.035, right=0.98, bottom=0.025, top=0.945, hspace=0.44, wspace=0.18
    )
    figure.savefig(args.output)
    plt.close(figure)


if __name__ == "__main__":
    main()
