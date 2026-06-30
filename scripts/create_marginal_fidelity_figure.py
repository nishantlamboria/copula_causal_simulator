from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

BENCHMARK_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "benchmarks"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "progress_presentation"
    / "figures"
)

BENCHMARK_SEED = 42


# ---------------------------------------------------------------------
# Dataset order and display names
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


def load_variable_level_errors() -> pd.DataFrame:
    """
    Collect one marginal-fidelity value per variable and graph.

    Each benchmark evaluation contains:

        marginal_error_summary.csv

    For every variable, the plotted value is the median normalized
    absolute quantile error across the five evaluated quantiles:

        0.05, 0.25, 0.50, 0.75, 0.95.

    Values are pooled over the three random benchmark graphs.
    """
    frames: list[pd.DataFrame] = []

    for dataset_id in DATASET_ORDER:
        for indegree in (1, 2):
            benchmark_dir = (
                BENCHMARK_ROOT
                / dataset_id
                / f"indegree{indegree}"
                / f"seed_{BENCHMARK_SEED}"
            )

            summary_paths = sorted(
                benchmark_dir.glob(
                    "graph_*/evaluation/marginal_error_summary.csv"
                )
            )

            if not summary_paths:
                raise FileNotFoundError(
                    "No marginal-error summaries were found in:\n"
                    f"{benchmark_dir}\n"
                    "Run the corresponding benchmark first."
                )

            for path in summary_paths:
                frame = pd.read_csv(path)

                required_columns = {
                    "variable",
                    "median_normalized_abs_quantile_error",
                }

                missing_columns = required_columns.difference(
                    frame.columns
                )

                if missing_columns:
                    raise ValueError(
                        f"{path} is missing required columns: "
                        f"{sorted(missing_columns)}"
                    )

                selected = frame[
                    [
                        "variable",
                        "median_normalized_abs_quantile_error",
                    ]
                ].copy()

                selected = selected.rename(
                    columns={
                        "median_normalized_abs_quantile_error":
                            "marginal_error"
                    }
                )

                selected["dataset_id"] = dataset_id
                selected["dataset_label"] = (
                    DATASET_LABELS[dataset_id]
                )
                selected["indegree"] = indegree

                # path:
                # graph_0000/evaluation/marginal_error_summary.csv
                selected["graph_id"] = path.parents[1].name

                frames.append(selected)

    result = pd.concat(
        frames,
        axis=0,
        ignore_index=True,
    )

    result["marginal_error"] = pd.to_numeric(
        result["marginal_error"],
        errors="coerce",
    )

    result = result.dropna(
        subset=["marginal_error"]
    )

    if result.empty:
        raise ValueError(
            "No valid marginal-error values were collected."
        )

    return result


def configure_matplotlib() -> None:
    """Apply presentation-compatible plotting defaults."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.labelcolor": INK,
            "axes.edgecolor": RULE_GREY,
            "axes.linewidth": 0.9,
            "xtick.color": INK,
            "ytick.color": SLATE,
            "text.color": INK,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
        }
    )


def draw_boxplot(
    axis: Axes,
    values: np.ndarray,
    position: float,
    fill_color: str,
    edge_color: str,
) -> None:
    """Draw one styled boxplot."""
    artists = axis.boxplot(
        [values],
        positions=[position],
        widths=0.28,
        patch_artist=True,
        showfliers=False,
        whis=(5, 95),
        manage_ticks=False,
        boxprops={
            "edgecolor": edge_color,
            "linewidth": 1.6,
        },
        whiskerprops={
            "color": edge_color,
            "linewidth": 1.35,
        },
        capprops={
            "color": edge_color,
            "linewidth": 1.35,
        },
        medianprops={
            "color": edge_color,
            "linewidth": 2.3,
        },
    )

    artists["boxes"][0].set_facecolor(
        fill_color
    )


def create_figure(
    data: pd.DataFrame,
) -> tuple[Path, Path]:
    """Create and save the presentation figure."""
    configure_matplotlib()

    figure, axis = plt.subplots(
        figsize=(12.0, 5.6)
    )

    random_generator = np.random.default_rng(
        BENCHMARK_SEED
    )

    group_centres = np.arange(
        len(DATASET_ORDER),
        dtype=float,
    )

    offsets = {
        1: -0.19,
        2: 0.19,
    }

    styles = {
        1: {
            "fill": SOFT_TEAL,
            "edge": DEEP_TEAL,
            "point": MUTED_TEAL,
        },
        2: {
            "fill": SOFT_ORANGE,
            "edge": ACCENT_ORANGE,
            "point": ACCENT_ORANGE,
        },
    }

    for dataset_index, dataset_id in enumerate(
        DATASET_ORDER
    ):
        for indegree in (1, 2):
            values = data.loc[
                (
                    data["dataset_id"]
                    == dataset_id
                )
                & (
                    data["indegree"]
                    == indegree
                ),
                "marginal_error",
            ].to_numpy(dtype=float)

            if values.size == 0:
                raise ValueError(
                    "No values found for "
                    f"dataset={dataset_id}, "
                    f"indegree={indegree}."
                )

            position = (
                group_centres[dataset_index]
                + offsets[indegree]
            )

            style = styles[indegree]

            # Slight horizontal displacement makes individual
            # variables visible without obscuring the boxplot.
            jitter = random_generator.normal(
                loc=0.0,
                scale=0.032,
                size=values.size,
            )

            axis.scatter(
                position + jitter,
                values,
                s=20,
                color=style["point"],
                alpha=0.38,
                linewidths=0,
                zorder=2,
            )

            draw_boxplot(
                axis=axis,
                values=values,
                position=position,
                fill_color=style["fill"],
                edge_color=style["edge"],
            )

    maximum_error = float(
        data["marginal_error"].max()
    )

    axis.set_ylim(
        0.0,
        maximum_error * 1.17,
    )

    axis.set_xticks(
        group_centres
    )

    axis.set_xticklabels(
        [
            DATASET_LABELS[dataset_id]
            for dataset_id in DATASET_ORDER
        ],
        fontsize=12,
        fontweight="semibold",
    )

    axis.set_ylabel(
        "Median normalized quantile error",
        labelpad=10,
        fontweight="semibold",
    )

    # axis.text(
    #     0.995,
    #     0.975,
    #     "Lower is better",
    #     transform=axis.transAxes,
    #     ha="right",
    #     va="top",
    #     fontsize=10.5,
    #     fontweight="semibold",
    #     color=DEEP_TEAL,
    # )

    axis.grid(
        axis="y",
        color=LIGHT_GREY,
        linewidth=1.0,
        alpha=1.0,
    )

    axis.set_axisbelow(True)

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(RULE_GREY)
    axis.spines["bottom"].set_color(RULE_GREY)

    axis.tick_params(
        axis="x",
        length=0,
        pad=9,
    )

    axis.tick_params(
        axis="y",
        labelsize=10.5,
    )

    legend_handles = [
        Patch(
            facecolor=SOFT_TEAL,
            edgecolor=DEEP_TEAL,
            linewidth=1.5,
            label="Max indegree 1",
        ),
        Patch(
            facecolor=SOFT_ORANGE,
            edgecolor=ACCENT_ORANGE,
            linewidth=1.5,
            label="Max indegree 2",
        ),
    ]

    axis.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=False,
        ncol=2,
        fontsize=10.5,
        handlelength=1.6,
        columnspacing=1.8,
    )

    # figure.text(
    #     0.50,
    #     0.025,
    #     (
    #         "Each point represents one variable in one random graph; "
    #         "the value is the median error over quantiles "
    #         "0.05, 0.25, 0.50, 0.75 and 0.95."
    #     ),
    #     ha="center",
    #     va="bottom",
    #     fontsize=9.5,
    #     color=SLATE,
    # )

    figure.subplots_adjust(
        left=0.095,
        right=0.985,
        top=0.95,
        bottom=0.18,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf_path = (
        OUTPUT_DIR
        / "marginal_fidelity_by_dataset.pdf"
    )

    png_path = (
        OUTPUT_DIR
        / "marginal_fidelity_by_dataset.png"
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
    data = load_variable_level_errors()

    pdf_path, png_path = create_figure(
        data
    )

    print("Marginal-fidelity figure created:")
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()