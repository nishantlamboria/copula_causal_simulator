"""Analyze synthetic pair-copula family-recovery results.

This script produces:

1. count and row-normalized confusion matrices;
2. recovery summaries by family, tau, sample size, and Student df;
3. detailed Student-t and Frank diagnostics;
4. a table containing every misclassified run.

Run from the project root:

    python scripts/synthetic_validation/analyze_pair_recovery.py ^
        --results outputs/synthetic_validation/pair_family_recovery_intermediate/pair_family_recovery_results.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze pair-copula family-recovery results."
    )

    parser.add_argument(
        "--results",
        required=True,
        type=str,
        help="Path to pair_family_recovery_results.csv.",
    )

    parser.add_argument(
        "--output-directory",
        default=None,
        type=str,
        help=(
            "Optional output directory. By default, an analysis "
            "directory is created beside the input CSV."
        ),
    )

    return parser.parse_args()


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def coerce_boolean_column(
    dataframe: pd.DataFrame,
    column: str,
) -> None:
    """Convert CSV boolean representations into actual booleans."""
    if column not in dataframe.columns:
        return

    if pd.api.types.is_bool_dtype(dataframe[column]):
        return

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }

    converted = (
        dataframe[column]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )

    if converted.isna().any():
        invalid_values = sorted(
            dataframe.loc[
                converted.isna(),
                column,
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Could not interpret values in {column!r} as booleans: "
            f"{invalid_values}"
        )

    dataframe[column] = converted.astype(bool)


def validate_columns(dataframe: pd.DataFrame) -> None:
    required_columns = {
        "run_id",
        "status",
        "true_family",
        "selected_family",
        "family_correct",
        "true_tau",
        "estimated_tau",
        "tau_abs_error",
        "sample_size",
        "seed",
        "true_student_df",
        "estimated_student_df",
        "student_df_abs_error",
    }

    missing = required_columns.difference(dataframe.columns)

    if missing:
        raise ValueError(
            "The results file is missing required columns: "
            + ", ".join(sorted(missing))
        )


def create_selection_breakdown(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize which family was selected in each setting."""
    grouping_columns = [
        "true_family",
        "true_tau",
        "true_student_df",
        "sample_size",
        "selected_family",
        "selected_rotation",
    ]

    breakdown = (
        results
        .groupby(
            grouping_columns,
            dropna=False,
            as_index=False,
        )
        .agg(
            selection_count=("run_id", "count"),
        )
    )

    setting_totals = (
        breakdown
        .groupby(
            [
                "true_family",
                "true_tau",
                "true_student_df",
                "sample_size",
            ],
            dropna=False,
        )["selection_count"]
        .transform("sum")
    )

    breakdown["selection_proportion"] = (
        breakdown["selection_count"]
        / setting_totals
    )

    return breakdown.sort_values(
        [
            "true_family",
            "true_tau",
            "true_student_df",
            "sample_size",
            "selection_count",
        ],
        ascending=[
            True,
            True,
            True,
            True,
            False,
        ],
        na_position="last",
    ).reset_index(drop=True)


def create_setting_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize recovery for each complete generating setting."""
    return (
        results
        .groupby(
            [
                "true_family",
                "true_tau",
                "true_student_df",
                "sample_size",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            number_runs=("run_id", "count"),
            recovery_accuracy=("family_correct", "mean"),
            mean_tau_abs_error=("tau_abs_error", "mean"),
            median_tau_abs_error=("tau_abs_error", "median"),
            mean_estimated_tau=("estimated_tau", "mean"),
            mean_student_df_abs_error=(
                "student_df_abs_error",
                "mean",
            ),
            median_student_df_abs_error=(
                "student_df_abs_error",
                "median",
            ),
        )
        .sort_values(
            [
                "true_family",
                "true_tau",
                "true_student_df",
                "sample_size",
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )


def create_student_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    student_results = results.loc[
        results["true_family"] == "student"
    ].copy()

    if student_results.empty:
        return pd.DataFrame()

    student_results["selected_student"] = (
        student_results["selected_family"] == "student"
    )

    return (
        student_results
        .groupby(
            [
                "true_tau",
                "true_student_df",
                "sample_size",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            number_runs=("run_id", "count"),
            student_recovery_accuracy=(
                "selected_student",
                "mean",
            ),
            mean_tau_abs_error=("tau_abs_error", "mean"),
            median_tau_abs_error=("tau_abs_error", "median"),
            mean_estimated_student_df=(
                "estimated_student_df",
                "mean",
            ),
            mean_student_df_abs_error=(
                "student_df_abs_error",
                "mean",
            ),
            median_student_df_abs_error=(
                "student_df_abs_error",
                "median",
            ),
        )
        .sort_values(
            [
                "true_student_df",
                "true_tau",
                "sample_size",
            ]
        )
        .reset_index(drop=True)
    )


def create_frank_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    frank_results = results.loc[
        results["true_family"] == "frank"
    ].copy()

    if frank_results.empty:
        return pd.DataFrame()

    return (
        frank_results
        .groupby(
            [
                "true_tau",
                "sample_size",
            ],
            as_index=False,
        )
        .agg(
            number_runs=("run_id", "count"),
            recovery_accuracy=("family_correct", "mean"),
            mean_tau_abs_error=("tau_abs_error", "mean"),
            median_tau_abs_error=("tau_abs_error", "median"),
        )
        .sort_values(
            [
                "true_tau",
                "sample_size",
            ]
        )
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()

    results_path = resolve_project_path(args.results)

    if not results_path.exists():
        raise FileNotFoundError(
            f"Results file does not exist: {results_path}"
        )

    if args.output_directory is None:
        output_directory = (
            results_path.parent
            / "analysis"
        )
    else:
        output_directory = resolve_project_path(
            args.output_directory
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = pd.read_csv(results_path)

    validate_columns(results)

    coerce_boolean_column(
        results,
        "family_correct",
    )

    successful_results = results.loc[
        results["status"] == "success"
    ].copy()

    if successful_results.empty:
        raise RuntimeError(
            "The results file contains no successful runs."
        )

    confusion_counts = pd.crosstab(
        successful_results["true_family"],
        successful_results["selected_family"],
        rownames=["true_family"],
        colnames=["selected_family"],
        dropna=False,
    )

    confusion_proportions = pd.crosstab(
        successful_results["true_family"],
        successful_results["selected_family"],
        rownames=["true_family"],
        colnames=["selected_family"],
        normalize="index",
        dropna=False,
    )

    setting_summary = create_setting_summary(
        successful_results
    )

    selection_breakdown = create_selection_breakdown(
        successful_results
    )

    student_summary = create_student_summary(
        successful_results
    )

    frank_summary = create_frank_summary(
        successful_results
    )

    misclassified = successful_results.loc[
        ~successful_results["family_correct"]
    ].copy()

    misclassified = misclassified.sort_values(
        [
            "true_family",
            "true_tau",
            "true_student_df",
            "sample_size",
            "seed",
        ],
        na_position="last",
    )

    confusion_counts.to_csv(
        output_directory
        / "confusion_matrix_counts.csv"
    )

    confusion_proportions.to_csv(
        output_directory
        / "confusion_matrix_row_proportions.csv"
    )

    setting_summary.to_csv(
        output_directory
        / "recovery_by_setting.csv",
        index=False,
    )

    selection_breakdown.to_csv(
        output_directory
        / "selection_breakdown.csv",
        index=False,
    )

    student_summary.to_csv(
        output_directory
        / "student_recovery_summary.csv",
        index=False,
    )

    frank_summary.to_csv(
        output_directory
        / "frank_recovery_summary.csv",
        index=False,
    )

    misclassified.to_csv(
        output_directory
        / "misclassified_runs.csv",
        index=False,
    )

    print("=" * 80)
    print("Confusion matrix: counts")
    print("=" * 80)
    print(confusion_counts.to_string())

    print()
    print("=" * 80)
    print("Confusion matrix: row proportions")
    print("=" * 80)
    print(
        confusion_proportions
        .round(3)
        .to_string()
    )

    print()
    print("=" * 80)
    print("Student-t recovery")
    print("=" * 80)

    if student_summary.empty:
        print("No Student-t scenarios were found.")
    else:
        print(
            student_summary
            .round(4)
            .to_string(index=False)
        )

    print()
    print("=" * 80)
    print("Frank recovery")
    print("=" * 80)

    if frank_summary.empty:
        print("No Frank scenarios were found.")
    else:
        print(
            frank_summary
            .round(4)
            .to_string(index=False)
        )

    print()
    print("=" * 80)
    print("Misclassification counts")
    print("=" * 80)

    if misclassified.empty:
        print("No misclassifications.")
    else:
        counts = (
            misclassified
            .groupby(
                [
                    "true_family",
                    "selected_family",
                    "sample_size",
                ],
                as_index=False,
            )
            .agg(
                count=("run_id", "count")
            )
            .sort_values(
                [
                    "true_family",
                    "sample_size",
                    "count",
                ],
                ascending=[
                    True,
                    True,
                    False,
                ],
            )
        )

        print(counts.to_string(index=False))

    print()
    print(f"Analysis written to: {output_directory}")


if __name__ == "__main__":
    main()