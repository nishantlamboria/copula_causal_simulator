from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "benchmarks"
    / "combined"
    / "combined_graph_level_summary.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "progress_presentation"
    / "figures"
)


# ---------------------------------------------------------------------
# Dataset order and labels
# ---------------------------------------------------------------------
DATASET_ORDER = [
    "sachs",
    "diabetes",
    "breast_cancer",
]

DATASET_LABELS = {
    "sachs": "Sachs",
    "diabetes": "Diabetes",
    "breast_cancer": "Breast Cancer",
}


# ---------------------------------------------------------------------
# Presentation palette
# ---------------------------------------------------------------------
DEEP_TEAL = "#157F7A"
MUTED_TEAL = "#76B7B2"
SOFT_TEAL = "#DCEFED"

ACCENT_ORANGE = "#DE7C4D"
SOFT_ORANGE = "#F7E3D6"

INK = "#1F2933"
SLATE = "#66727F"
RULE_GREY = "#D8DEE3"
LIGHT_GREY = "#EDF1F3"
WHITE = "#FFFFFF"


REQUIRED_COLUMNS = {
    "dataset_id",
    "indegree",
    "graph_id",
    "n_samples",
    "generation_runtime_seconds",
    "generation_samples_per_second",
}


def load_runtime_data() -> pd.DataFrame:
    """Load graph-level runtime measurements from benchmark outputs."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "Benchmark graph-level summary not found:\n"
            f"{INPUT_PATH}\n"
            "Run scripts/collect_benchmark_summaries.py first."
        )

    frame = pd.read_csv(INPUT_PATH)

    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(
            "The graph-level benchmark table is missing required columns: "
            f"{sorted(missing)}"
        )

    frame = frame.copy()

    for column in [
        "indegree",
        "n_samples",
        "generation_runtime_seconds",
        "generation_samples_per_second",
    ]:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame = frame.dropna(
        subset=[
            "dataset_id",
            "indegree",
            "generation_runtime_seconds",
            "generation_samples_per_second",
        ]
    )

    frame["indegree"] = frame["indegree"].astype(int)

    frame = frame[
        frame["dataset_id"].isin(DATASET_ORDER)
        & frame["indegree"].isin([1, 2])
    ].copy()

    if frame.empty:
        raise ValueError(
            "No benchmark rows remain after filtering to the expected "
            "datasets and indegree settings."
        )

    expected_groups = {
        (dataset_id, indegree)
        for dataset_id in DATASET_ORDER
        for indegree in (1, 2)
    }

    available_groups = set(
        zip(frame["dataset_id"], frame["indegree"])
    )

    missing_groups = expected_groups.difference(available_groups)

    if missing_groups:
        formatted = ", ".join(
            f"{dataset_id}, indegree {indegree}"
            for dataset_id, indegree in sorted(missing_groups)
        )

        raise ValueError(
            f"Missing benchmark groups: {formatted}"
        )

    return frame


def configure_matplotlib() -> None:
    """Apply plotting defaults matching the presentation theme."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.labelcolor": INK,
            "axes.edgecolor": RULE_GREY,
            "axes.linewidth": 0.9,
            "xtick.color": SLATE,
            "ytick.color": INK,
            "text.color": INK,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
        }
    )


def create_figure(
    frame: pd.DataFrame,
) -> tuple[Path, Path]:
    """
    Create a presentation-ready runtime/throughput figure.

    Throughput is plotted on the horizontal axis. Since every graph
    generates 1000 observations, runtime and throughput contain the
    same information. Mean runtime is therefore annotated beside each
    mean throughput marker rather than plotted on a second axis.
    """
    configure_matplotlib()

    figure, axis = plt.subplots(
        figsize=(11.8, 5.8)
    )

    dataset_y = {
        dataset_id: float(
            len(DATASET_ORDER) - 1 - index
        )
        for index, dataset_id
        in enumerate(DATASET_ORDER)
    }

    y_offsets = {
        1: 0.13,
        2: -0.13,
    }

    styles = {
        1: {
            "edge": DEEP_TEAL,
            "fill": SOFT_TEAL,
            "raw": MUTED_TEAL,
        },
        2: {
            "edge": ACCENT_ORANGE,
            "fill": SOFT_ORANGE,
            "raw": ACCENT_ORANGE,
        },
    }

    all_throughputs = frame[
        "generation_samples_per_second"
    ].to_numpy(dtype=float)

    x_min = float(np.min(all_throughputs))
    x_max = float(np.max(all_throughputs))
    x_span = max(x_max - x_min, 1.0)

    # Light horizontal bands separate the datasets.
    for dataset_id in DATASET_ORDER:
        y = dataset_y[dataset_id]

        axis.axhspan(
            y - 0.34,
            y + 0.34,
            color=LIGHT_GREY,
            alpha=0.35,
            zorder=0,
        )

    for dataset_id in DATASET_ORDER:
        mean_points: dict[int, tuple[float, float]] = {}

        for indegree in (1, 2):
            selected = frame[
                (frame["dataset_id"] == dataset_id)
                & (frame["indegree"] == indegree)
            ]

            throughput = selected[
                "generation_samples_per_second"
            ].to_numpy(dtype=float)

            runtime = selected[
                "generation_runtime_seconds"
            ].to_numpy(dtype=float)

            mean_throughput = float(
                np.mean(throughput)
            )

            std_throughput = float(
                np.std(throughput, ddof=1)
            )

            mean_runtime = float(
                np.mean(runtime)
            )

            y = (
                dataset_y[dataset_id]
                + y_offsets[indegree]
            )

            style = styles[indegree]

            # Individual random-DAG benchmark repetitions.
            raw_y = np.full_like(
                throughput,
                y,
                dtype=float,
            )

            axis.scatter(
                throughput,
                raw_y,
                s=34,
                color=style["raw"],
                alpha=0.30,
                linewidths=0,
                zorder=3,
            )

            # Mean and standard deviation across the three graphs.
            axis.errorbar(
                mean_throughput,
                y,
                xerr=std_throughput,
                fmt="o",
                markersize=8.5,
                markerfacecolor=style["fill"],
                markeredgecolor=style["edge"],
                markeredgewidth=1.8,
                ecolor=style["edge"],
                elinewidth=1.5,
                capsize=4,
                capthick=1.4,
                zorder=5,
            )

            mean_points[indegree] = (
                mean_throughput,
                y,
            )

            text_offset = (
                8 if indegree == 1 else -8
            )

            vertical_alignment = (
                "bottom"
                if indegree == 1
                else "top"
            )

            axis.annotate(
                f"{mean_runtime:.2f} s",
                xy=(mean_throughput, y),
                xytext=(0, text_offset),
                textcoords="offset points",
                ha="center",
                va=vertical_alignment,
                fontsize=9.3,
                fontweight="semibold",
                color=style["edge"],
                zorder=6,
            )

        # Connect indegree-one and indegree-two means.
        x1, y1 = mean_points[1]
        x2, y2 = mean_points[2]

        axis.plot(
            [x1, x2],
            [y1, y2],
            color=RULE_GREY,
            linewidth=1.6,
            zorder=2,
        )

    axis.set_yticks(
        [
            dataset_y[dataset_id]
            for dataset_id in DATASET_ORDER
        ]
    )

    axis.set_yticklabels(
        [
            DATASET_LABELS[dataset_id]
            for dataset_id in DATASET_ORDER
        ],
        fontsize=12,
        fontweight="semibold",
    )

    axis.set_xlabel(
        "Generation throughput (samples per second)",
        labelpad=12,
        fontweight="semibold",
    )

    axis.set_xlim(
        x_min - 0.10 * x_span,
        x_max + 0.10 * x_span,
    )

    axis.set_ylim(
        -0.55,
        len(DATASET_ORDER) - 0.45,
    )

    # axis.text(
    #     0.995,
    #     0.975,
    #     "Higher is better",
    #     transform=axis.transAxes,
    #     ha="right",
    #     va="top",
    #     fontsize=10.5,
    #     fontweight="semibold",
    #     color=DEEP_TEAL,
    # )

    # axis.text(
    #     0.0,
    #     1.03,
    #     (
    #         "Large markers show means; horizontal bars show "
    #         "±1 SD across three random DAGs."
    #     ),
    #     transform=axis.transAxes,
    #     ha="left",
    #     va="bottom",
    #     fontsize=10,
    #     color=SLATE,
    # )

    axis.grid(
        axis="x",
        color=LIGHT_GREY,
        linewidth=1.0,
        alpha=1.0,
    )

    axis.set_axisbelow(True)

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.spines["bottom"].set_color(RULE_GREY)

    axis.tick_params(
        axis="y",
        length=0,
        pad=11,
    )

    axis.tick_params(
        axis="x",
        labelsize=10.5,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=SOFT_TEAL,
            markeredgecolor=DEEP_TEAL,
            markeredgewidth=1.6,
            markersize=8,
            label="Max indegree 1",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=SOFT_ORANGE,
            markeredgecolor=ACCENT_ORANGE,
            markeredgewidth=1.6,
            markersize=8,
            label="Max indegree 2",
        ),
    ]

    axis.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.27),
        ncol=2,
        frameon=False,
        fontsize=10.5,
        handletextpad=0.7,
        columnspacing=2.2,
    )

    # figure.text(
    #     0.50,
    #     0.022,
    #     (
    #         "Runtime labels report mean seconds required "
    #         "to generate 1000 observations."
    #     ),
    #     ha="center",
    #     va="bottom",
    #     fontsize=9.5,
    #     color=SLATE,
    # )

    # figure.subplots_adjust(
    #     left=0.16,
    #     right=0.985,
    #     top=0.88,
    #     bottom=0.24,
    # )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf_path = (
        OUTPUT_DIR
        / "runtime_throughput_by_dataset.pdf"
    )

    png_path = (
        OUTPUT_DIR
        / "runtime_throughput_by_dataset.png"
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    figure.savefig(
        png_path,
        dpi=320,
        bbox_inches="tight",
    )

    plt.close(figure)

    return pdf_path, png_path


def main() -> None:
    frame = load_runtime_data()

    pdf_path, png_path = create_figure(
        frame
    )

    print("Runtime/throughput figure created:")
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()