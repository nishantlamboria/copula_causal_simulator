from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


CORE_METRICS = [
    "marginal_mean_normalized_abs_quantile_error",
    "marginal_median_normalized_abs_quantile_error",
    "marginal_max_normalized_abs_quantile_error",
    "kendall_mean_abs_error",
    "kendall_median_abs_error",
    "spearman_mean_abs_error",
    "spearman_median_abs_error",
    "kendall_edge_generated_mean_abs_value",
    "kendall_nonedge_generated_mean_abs_value",
    "kendall_generated_edge_nonedge_abs_gap",
    "kendall_edge_mean_abs_error",
    "kendall_nonedge_mean_abs_error",
    "spearman_edge_generated_mean_abs_value",
    "spearman_nonedge_generated_mean_abs_value",
    "spearman_generated_edge_nonedge_abs_gap",
    "spearman_edge_mean_abs_error",
    "spearman_nonedge_mean_abs_error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect and compare indegree-1 and indegree-2 benchmark summaries."
        )
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["sachs", "diabetes"],
        help="Dataset identifiers to include.",
    )

    parser.add_argument(
        "--indegrees",
        nargs="+",
        type=int,
        default=[1, 2],
        help="Generator indegrees to include.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Benchmark seed directory to collect, e.g. seed_42.",
    )

    parser.add_argument(
        "--benchmark-root",
        type=str,
        default="outputs/benchmarks",
        help="Root directory containing benchmark outputs.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/benchmarks/combined",
        help="Directory for combined machine-readable outputs.",
    )

    parser.add_argument(
        "--report-dir",
        type=str,
        default="reports/final_tables",
        help="Directory for report-ready CSV tables.",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any requested benchmark summary is missing.",
    )

    return parser.parse_args()


def sanitize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): sanitize_json_value(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [sanitize_json_value(item) for item in value]

    if isinstance(value, tuple):
        return [sanitize_json_value(item) for item in value]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return None
        return value

    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value

    if pd.isna(value):
        return None

    return value


def save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            sanitize_json_value(payload),
            file,
            indent=2,
        )


def benchmark_summary_path(
    benchmark_root: Path,
    dataset_id: str,
    indegree: int,
    seed: int,
) -> Path:
    return (
        benchmark_root
        / dataset_id
        / f"indegree{indegree}"
        / f"seed_{seed}"
        / "benchmark_summary.csv"
    )


def load_one_summary(
    path: Path,
    dataset_id: str,
    indegree: int,
    benchmark_seed: int,
) -> pd.DataFrame:
    frame = pd.read_csv(path)

    if frame.empty:
        raise ValueError(f"Benchmark summary is empty: {path}")

    frame = frame.copy()

    frame["dataset_id"] = dataset_id
    frame["indegree"] = indegree
    frame["benchmark_seed"] = benchmark_seed
    frame["source_summary_file"] = str(path.relative_to(PROJECT_ROOT))

    if "generator" not in frame.columns:
        frame["generator"] = (
            "pair_copula_indegree_one"
            if indegree == 1
            else "conditional_dvine_indegree_two"
        )

    return frame


def collect_summaries(
    benchmark_root: Path,
    datasets: list[str],
    indegrees: list[int],
    seed: int,
    strict: bool,
) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    missing_files: list[str] = []

    for dataset_id in datasets:
        for indegree in indegrees:
            path = benchmark_summary_path(
                benchmark_root=benchmark_root,
                dataset_id=dataset_id,
                indegree=indegree,
                seed=seed,
            )

            if not path.exists():
                message = f"Missing benchmark summary: {path}"

                if strict:
                    raise FileNotFoundError(message)

                print(f"Warning: {message}")
                missing_files.append(str(path))
                continue

            print(f"Loading: {path}")

            frame = load_one_summary(
                path=path,
                dataset_id=dataset_id,
                indegree=indegree,
                benchmark_seed=seed,
            )

            frames.append(frame)

    if not frames:
        raise FileNotFoundError(
            "No benchmark summaries were found for the requested datasets, "
            "indegrees, and seed."
        )

    combined = pd.concat(
        frames,
        axis=0,
        ignore_index=True,
        sort=False,
    )

    sort_columns = [
        column
        for column in ["dataset_id", "indegree", "graph_id"]
        if column in combined.columns
    ]

    combined = combined.sort_values(sort_columns).reset_index(drop=True)

    return combined, missing_files


def available_metrics(frame: pd.DataFrame) -> list[str]:
    metrics = []

    for metric in CORE_METRICS:
        if metric not in frame.columns:
            continue

        numeric_series = pd.to_numeric(frame[metric], errors="coerce")

        if numeric_series.notna().any():
            metrics.append(metric)

    return metrics


def build_aggregate_table(
    graph_level: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    group_columns = ["dataset_id", "indegree"]

    base_aggregation: dict[str, tuple[str, str]] = {}

    if "graph_id" in graph_level.columns:
        base_aggregation["num_graphs"] = ("graph_id", "count")

    if "num_edges" in graph_level.columns:
        base_aggregation["mean_num_edges"] = ("num_edges", "mean")
        base_aggregation["std_num_edges"] = ("num_edges", "std")

    if "n_samples" in graph_level.columns:
        base_aggregation["mean_n_samples"] = ("n_samples", "mean")

    if base_aggregation:
        aggregate = (
            graph_level
            .groupby(group_columns, dropna=False)
            .agg(**base_aggregation)
            .reset_index()
        )
    else:
        aggregate = (
            graph_level[group_columns]
            .drop_duplicates()
            .reset_index(drop=True)
        )

    for metric in metrics:
        metric_values = graph_level[
            group_columns + [metric]
        ].copy()

        metric_values[metric] = pd.to_numeric(
            metric_values[metric],
            errors="coerce",
        )

        metric_summary = (
            metric_values
            .groupby(group_columns, dropna=False)[metric]
            .agg(["mean", "std", "median", "min", "max"])
            .reset_index()
            .rename(
                columns={
                    "mean": f"{metric}__mean",
                    "std": f"{metric}__std",
                    "median": f"{metric}__median",
                    "min": f"{metric}__min",
                    "max": f"{metric}__max",
                }
            )
        )

        aggregate = aggregate.merge(
            metric_summary,
            on=group_columns,
            how="outer",
        )

    return aggregate.sort_values(group_columns).reset_index(drop=True)


def build_report_table(
    aggregate: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    preferred_metrics = [
        "marginal_median_normalized_abs_quantile_error",
        "kendall_mean_abs_error",
        "spearman_mean_abs_error",
        "kendall_edge_generated_mean_abs_value",
        "kendall_nonedge_generated_mean_abs_value",
        "kendall_generated_edge_nonedge_abs_gap",
        "spearman_edge_generated_mean_abs_value",
        "spearman_nonedge_generated_mean_abs_value",
        "spearman_generated_edge_nonedge_abs_gap",
    ]

    selected_metrics = [
        metric for metric in preferred_metrics if metric in metrics
    ]

    columns = [
        column
        for column in [
            "dataset_id",
            "indegree",
            "num_graphs",
            "mean_num_edges",
        ]
        if column in aggregate.columns
    ]

    for metric in selected_metrics:
        mean_column = f"{metric}__mean"
        std_column = f"{metric}__std"

        if mean_column in aggregate.columns:
            columns.append(mean_column)

        if std_column in aggregate.columns:
            columns.append(std_column)

    return aggregate[columns].copy()


def build_indegree_comparison(
    aggregate: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    """
    Produce one row per dataset with indegree-1 mean, indegree-2 mean,
    and delta = indegree2 - indegree1.
    """
    mean_metrics = [
        f"{metric}__mean"
        for metric in metrics
        if f"{metric}__mean" in aggregate.columns
    ]

    rows: list[dict[str, Any]] = []

    for dataset_id, dataset_frame in aggregate.groupby("dataset_id"):
        row: dict[str, Any] = {
            "dataset_id": dataset_id,
        }

        available_indegrees = set(
            dataset_frame["indegree"].astype(int).tolist()
        )

        row["has_indegree1"] = 1 in available_indegrees
        row["has_indegree2"] = 2 in available_indegrees

        indegree1 = dataset_frame[
            dataset_frame["indegree"] == 1
        ]

        indegree2 = dataset_frame[
            dataset_frame["indegree"] == 2
        ]

        for mean_metric in mean_metrics:
            base_metric = mean_metric.removesuffix("__mean")

            value_1 = (
                float(indegree1.iloc[0][mean_metric])
                if not indegree1.empty
                and pd.notna(indegree1.iloc[0][mean_metric])
                else np.nan
            )

            value_2 = (
                float(indegree2.iloc[0][mean_metric])
                if not indegree2.empty
                and pd.notna(indegree2.iloc[0][mean_metric])
                else np.nan
            )

            row[f"{base_metric}__indegree1"] = value_1
            row[f"{base_metric}__indegree2"] = value_2

            if pd.notna(value_1) and pd.notna(value_2):
                row[f"{base_metric}__delta_2_minus_1"] = (
                    value_2 - value_1
                )
            else:
                row[f"{base_metric}__delta_2_minus_1"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows).sort_values("dataset_id").reset_index(drop=True)


def print_compact_summary(report_table: pd.DataFrame) -> None:
    display_columns = [
        "dataset_id",
        "indegree",
        "num_graphs",
        "mean_num_edges",
        "marginal_median_normalized_abs_quantile_error__mean",
        "kendall_edge_generated_mean_abs_value__mean",
        "kendall_nonedge_generated_mean_abs_value__mean",
        "kendall_generated_edge_nonedge_abs_gap__mean",
        "spearman_generated_edge_nonedge_abs_gap__mean",
    ]

    display_columns = [
        column
        for column in display_columns
        if column in report_table.columns
    ]

    print("\nCombined benchmark overview:")

    if display_columns:
        print(
            report_table[display_columns].to_string(
                index=False,
                float_format=lambda value: f"{value:.6f}",
            )
        )
    else:
        print(report_table.to_string(index=False))


def main() -> None:
    args = parse_args()

    benchmark_root = PROJECT_ROOT / args.benchmark_root
    output_dir = PROJECT_ROOT / args.output_dir
    report_dir = PROJECT_ROOT / args.report_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    graph_level, missing_files = collect_summaries(
        benchmark_root=benchmark_root,
        datasets=args.datasets,
        indegrees=args.indegrees,
        seed=args.seed,
        strict=args.strict,
    )

    metrics = available_metrics(graph_level)

    if not metrics:
        raise ValueError(
            "No recognized numeric benchmark metrics were found."
        )

    aggregate = build_aggregate_table(
        graph_level=graph_level,
        metrics=metrics,
    )

    report_table = build_report_table(
        aggregate=aggregate,
        metrics=metrics,
    )

    indegree_comparison = build_indegree_comparison(
        aggregate=aggregate,
        metrics=metrics,
    )

    graph_level_path = output_dir / "combined_graph_level_summary.csv"
    aggregate_path = output_dir / "combined_aggregate_summary.csv"
    comparison_path = output_dir / "indegree_comparison.csv"
    manifest_path = output_dir / "collection_manifest.json"

    report_summary_path = report_dir / "benchmark_summary_by_dataset_and_indegree.csv"
    report_comparison_path = report_dir / "indegree1_vs_indegree2_comparison.csv"

    graph_level.to_csv(graph_level_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    indegree_comparison.to_csv(comparison_path, index=False)

    report_table.to_csv(report_summary_path, index=False)
    indegree_comparison.to_csv(report_comparison_path, index=False)

    manifest = {
        "datasets_requested": args.datasets,
        "indegrees_requested": args.indegrees,
        "benchmark_seed": args.seed,
        "num_graph_level_rows": int(len(graph_level)),
        "metrics_collected": metrics,
        "missing_files": missing_files,
        "outputs": {
            "graph_level_summary": str(
                graph_level_path.relative_to(PROJECT_ROOT)
            ),
            "aggregate_summary": str(
                aggregate_path.relative_to(PROJECT_ROOT)
            ),
            "indegree_comparison": str(
                comparison_path.relative_to(PROJECT_ROOT)
            ),
            "report_summary": str(
                report_summary_path.relative_to(PROJECT_ROOT)
            ),
            "report_comparison": str(
                report_comparison_path.relative_to(PROJECT_ROOT)
            ),
        },
    }

    save_json(manifest, manifest_path)

    print_compact_summary(report_table)

    print("\nCollection complete.")
    print(f"Graph-level table: {graph_level_path}")
    print(f"Aggregate table: {aggregate_path}")
    print(f"Indegree comparison: {comparison_path}")
    print(f"Report summary: {report_summary_path}")
    print(f"Report comparison: {report_comparison_path}")

    if missing_files:
        print("\nSome requested benchmark files were missing:")
        for missing_file in missing_files:
            print(f"  - {missing_file}")


if __name__ == "__main__":
    main()