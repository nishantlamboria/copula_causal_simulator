from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "benchmarks"
    / "combined"
    / "combined_aggregate_summary.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "progress_presentation"
    / "figures"
)

DATASET_ORDER = [
    "breast_cancer",
    "diabetes",
    "sachs",
]

DATASET_LABELS = {
    "breast_cancer": "Breast cancer",
    "diabetes": "Diabetes",
    "sachs": "Sachs",
}


def require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [
        column
        for column in columns
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            "The aggregate benchmark table is missing required columns: "
            f"{missing}"
        )


def metric_value(
    frame: pd.DataFrame,
    dataset_id: str,
    indegree: int,
    column: str,
) -> float:
    selected = frame[
        (frame["dataset_id"] == dataset_id)
        & (frame["indegree"] == indegree)
    ]

    if len(selected) != 1:
        raise ValueError(
            "Expected exactly one aggregate row for "
            f"dataset={dataset_id}, indegree={indegree}; "
            f"found {len(selected)}."
        )

    value = pd.to_numeric(
        selected.iloc[0][column],
        errors="coerce",
    )

    if pd.isna(value):
        raise ValueError(
            f"Metric {column} is missing for "
            f"dataset={dataset_id}, indegree={indegree}."
        )

    return float(value)


def create_tail_error_figure(frame: pd.DataFrame) -> None:
    lower_column = (
        "tail_lower_all_mean_abs_error__mean"
    )
    upper_column = (
        "tail_upper_all_mean_abs_error__mean"
    )

    categories: list[str] = []
    indegree1_values: list[float] = []
    indegree2_values: list[float] = []

    for dataset_id in DATASET_ORDER:
        for tail_name, column in [
            ("Lower tail", lower_column),
            ("Upper tail", upper_column),
        ]:
            categories.append(
                f"{DATASET_LABELS[dataset_id]}\n{tail_name}"
            )

            indegree1_values.append(
                metric_value(
                    frame,
                    dataset_id,
                    1,
                    column,
                )
            )

            indegree2_values.append(
                metric_value(
                    frame,
                    dataset_id,
                    2,
                    column,
                )
            )

    x = np.arange(len(categories))
    width = 0.36

    figure = plt.figure(figsize=(12, 6.5))
    axis = figure.add_subplot(111)

    bars_1 = axis.bar(
        x - width / 2,
        indegree1_values,
        width,
        label="Max indegree 1",
    )

    bars_2 = axis.bar(
        x + width / 2,
        indegree2_values,
        width,
        label="Max indegree 2",
    )

    axis.set_title(
        "Tail-dependence reconstruction error"
    )
    axis.set_ylabel(
        "Mean absolute error (lower is better)"
    )
    axis.set_xticks(x)
    axis.set_xticklabels(categories)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)

    for bars in (bars_1, bars_2):
        for bar in bars:
            height = bar.get_height()

            axis.annotate(
                f"{height:.3f}",
                xy=(
                    bar.get_x()
                    + bar.get_width() / 2,
                    height,
                ),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIR
        / "tail_dependence_error_by_dataset.png",
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        OUTPUT_DIR
        / "tail_dependence_error_by_dataset.pdf",
        bbox_inches="tight",
    )

    plt.close(figure)


def create_throughput_figure(
    frame: pd.DataFrame,
) -> None:
    throughput_column = (
        "generation_samples_per_second__mean"
    )

    labels = [
        DATASET_LABELS[dataset_id]
        for dataset_id in DATASET_ORDER
    ]

    indegree1_values = [
        metric_value(
            frame,
            dataset_id,
            1,
            throughput_column,
        )
        for dataset_id in DATASET_ORDER
    ]

    indegree2_values = [
        metric_value(
            frame,
            dataset_id,
            2,
            throughput_column,
        )
        for dataset_id in DATASET_ORDER
    ]

    x = np.arange(len(labels))
    width = 0.36

    figure = plt.figure(figsize=(9, 6))
    axis = figure.add_subplot(111)

    bars_1 = axis.bar(
        x - width / 2,
        indegree1_values,
        width,
        label="Max indegree 1",
    )

    bars_2 = axis.bar(
        x + width / 2,
        indegree2_values,
        width,
        label="Max indegree 2",
    )

    axis.set_title(
        "Synthetic-data generation throughput"
    )
    axis.set_ylabel(
        "Samples per second (higher is better)"
    )
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)

    maximum = max(
        indegree1_values + indegree2_values
    )
    axis.set_ylim(0, maximum * 1.16)

    for bars in (bars_1, bars_2):
        for bar in bars:
            height = bar.get_height()

            axis.annotate(
                f"{height:.0f}",
                xy=(
                    bar.get_x()
                    + bar.get_width() / 2,
                    height,
                ),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIR
        / "generation_throughput_by_dataset.png",
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        OUTPUT_DIR
        / "generation_throughput_by_dataset.pdf",
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Aggregate benchmark table not found: "
            f"{INPUT_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame = pd.read_csv(INPUT_PATH)

    required_columns = [
        "dataset_id",
        "indegree",
        "tail_lower_all_mean_abs_error__mean",
        "tail_upper_all_mean_abs_error__mean",
        "generation_samples_per_second__mean",
    ]

    require_columns(
        frame,
        required_columns,
    )

    create_tail_error_figure(frame)
    create_throughput_figure(frame)

    print("Presentation figures created:")
    print(
        OUTPUT_DIR
        / "tail_dependence_error_by_dataset.png"
    )
    print(
        OUTPUT_DIR
        / "generation_throughput_by_dataset.png"
    )


if __name__ == "__main__":
    main()
