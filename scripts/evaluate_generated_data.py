from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from copula_causal_sim.config import load_config
from copula_causal_sim.data.loaders import load_tabular_dataset
from copula_causal_sim.evaluation.realism import (
    compute_marginal_quantile_errors,
    summarize_marginal_errors,
    correlation_matrix,
    matrix_absolute_error,
    summarize_matrix_error,
    top_pairwise_errors,
    graph_pairwise_dependence_table,
    summarize_graph_dependence,
    tail_dependence_table,
    summarize_tail_dependence,
    save_dataframe,
    save_json,
)
from copula_causal_sim.graphs.dag import summarize_dag


DEFAULT_TAIL_PROBABILITY = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generated data against the calibration dataset."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Dataset config used to load the real calibration data.",
    )

    parser.add_argument(
        "--generated-x",
        type=str,
        required=True,
        help="Path to generated data CSV in original variable scale.",
    )

    parser.add_argument(
        "--graph",
        type=str,
        default=None,
        help="Optional adjacency matrix CSV used for generation.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where evaluation outputs will be written.",
    )

    parser.add_argument(
        "--tail-probability",
        type=float,
        default=DEFAULT_TAIL_PROBABILITY,
        help=(
            "Tail probability p used for empirical lower- and upper-tail "
            "dependence. Must satisfy 0 < p < 0.5. Default: 0.05."
        ),
    )

    return parser.parse_args()


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve relative paths from the project root."""
    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_graph(graph_path: str | None):
    if graph_path is None:
        return None

    path = resolve_project_path(graph_path)

    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")

    return pd.read_csv(path, header=None).to_numpy(dtype=int)


def make_empirical_pseudo_observations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert every variable to empirical copula coordinates.

    For n observations, average ranks are divided by n + 1. This keeps all
    pseudo-observations strictly inside (0, 1), handles ties deterministically,
    and makes tail-dependence comparisons invariant to marginal scales.
    """
    if df.empty:
        raise ValueError("Cannot construct pseudo-observations from empty data.")

    if df.isna().any().any():
        raise ValueError("Cannot construct pseudo-observations with missing values.")

    non_numeric = [
        column
        for column in df.columns
        if not pd.api.types.is_numeric_dtype(df[column])
    ]

    if non_numeric:
        raise ValueError(
            f"Cannot construct pseudo-observations from non-numeric columns: "
            f"{non_numeric}"
        )

    n_rows = len(df)

    pseudo = df.rank(
        axis=0,
        method="average",
        na_option="keep",
        pct=False,
    ) / float(n_rows + 1)

    pseudo = pseudo.astype(float)

    if pseudo.isna().any().any():
        raise RuntimeError("Pseudo-observation construction produced missing values.")

    return pseudo


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    dataset_id = config["dataset_id"]

    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating generated data for dataset: {dataset_id}")

    print("Loading real calibration data...")
    real_df = load_tabular_dataset(config)

    print("Loading generated data...")
    generated_path = resolve_project_path(args.generated_x)

    if not generated_path.exists():
        raise FileNotFoundError(
            f"Generated data file not found: {generated_path}"
        )

    generated_df = pd.read_csv(generated_path)

    print(f"Real data shape: {real_df.shape}")
    print(f"Generated data shape: {generated_df.shape}")

    if list(real_df.columns) != list(generated_df.columns):
        raise ValueError(
            "Column mismatch.\n"
            f"Real columns: {list(real_df.columns)}\n"
            f"Generated columns: {list(generated_df.columns)}"
        )

    # ------------------------------------------------------------------
    # 1. Marginal quantile evaluation
    # ------------------------------------------------------------------
    print("Computing marginal quantile errors...")

    marginal_errors = compute_marginal_quantile_errors(
        real_df=real_df,
        generated_df=generated_df,
    )

    marginal_summary = summarize_marginal_errors(marginal_errors)

    marginal_errors.to_csv(
        output_dir / "marginal_quantile_errors.csv",
        index=False,
    )

    marginal_summary.to_csv(
        output_dir / "marginal_error_summary.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 2. Kendall tau evaluation
    # ------------------------------------------------------------------
    print("Computing Kendall tau matrices...")

    kendall_real = correlation_matrix(real_df, method="kendall")
    kendall_generated = correlation_matrix(generated_df, method="kendall")
    kendall_error = matrix_absolute_error(kendall_real, kendall_generated)

    save_dataframe(kendall_real, output_dir / "kendall_real.csv")
    save_dataframe(kendall_generated, output_dir / "kendall_generated.csv")
    save_dataframe(kendall_error, output_dir / "kendall_abs_error.csv")

    kendall_top_errors = top_pairwise_errors(
        error_matrix=kendall_error,
        real_matrix=kendall_real,
        generated_matrix=kendall_generated,
        top_k=20,
    )

    kendall_top_errors.to_csv(
        output_dir / "kendall_top_pairwise_errors.csv",
        index=False,
    )

    kendall_summary = summarize_matrix_error(kendall_error)

    # ------------------------------------------------------------------
    # 3. Spearman rho evaluation
    # ------------------------------------------------------------------
    print("Computing Spearman rho matrices...")

    spearman_real = correlation_matrix(real_df, method="spearman")
    spearman_generated = correlation_matrix(generated_df, method="spearman")
    spearman_error = matrix_absolute_error(spearman_real, spearman_generated)

    save_dataframe(spearman_real, output_dir / "spearman_real.csv")
    save_dataframe(spearman_generated, output_dir / "spearman_generated.csv")
    save_dataframe(spearman_error, output_dir / "spearman_abs_error.csv")

    spearman_top_errors = top_pairwise_errors(
        error_matrix=spearman_error,
        real_matrix=spearman_real,
        generated_matrix=spearman_generated,
        top_k=20,
    )

    spearman_top_errors.to_csv(
        output_dir / "spearman_top_pairwise_errors.csv",
        index=False,
    )

    spearman_summary = summarize_matrix_error(spearman_error)

    # ------------------------------------------------------------------
    # 4. Graph metadata and graph-aware rank dependence
    # ------------------------------------------------------------------
    print("Reading graph metadata...")

    adjacency = load_graph(args.graph)
    graph_summary = None
    graph_dependence_summary = None

    if adjacency is not None:
        graph_summary = summarize_dag(adjacency)
        save_json(graph_summary, output_dir / "graph_summary.json")

        print("Computing graph-aware edge/non-edge dependence summaries...")

        kendall_graph_table = graph_pairwise_dependence_table(
            real_matrix=kendall_real,
            generated_matrix=kendall_generated,
            adjacency=adjacency,
        )

        spearman_graph_table = graph_pairwise_dependence_table(
            real_matrix=spearman_real,
            generated_matrix=spearman_generated,
            adjacency=adjacency,
        )

        kendall_graph_table.to_csv(
            output_dir / "kendall_graph_pairwise_dependence.csv",
            index=False,
        )

        spearman_graph_table.to_csv(
            output_dir / "spearman_graph_pairwise_dependence.csv",
            index=False,
        )

        graph_dependence_summary = {
            "kendall": summarize_graph_dependence(kendall_graph_table),
            "spearman": summarize_graph_dependence(spearman_graph_table),
        }

        save_json(
            graph_dependence_summary,
            output_dir / "graph_dependence_summary.json",
        )

    # ------------------------------------------------------------------
    # 5. Empirical lower- and upper-tail dependence
    # ------------------------------------------------------------------
    print(
        "Computing empirical tail dependence "
        f"with tail probability p={args.tail_probability:.4f}..."
    )

    real_u_df = make_empirical_pseudo_observations(real_df)
    generated_u_df = make_empirical_pseudo_observations(generated_df)

    tail_pairwise = tail_dependence_table(
        real_u_df=real_u_df,
        generated_u_df=generated_u_df,
        adjacency=adjacency,
        tail_probability=args.tail_probability,
    )

    tail_summary = summarize_tail_dependence(tail_pairwise)

    tail_pairwise.to_csv(
        output_dir / "tail_dependence_pairwise.csv",
        index=False,
    )

    save_json(
        tail_summary,
        output_dir / "tail_dependence_summary.json",
    )

    lower_tail_top_errors = tail_pairwise.sort_values(
        "lower_tail_absolute_error",
        ascending=False,
    ).head(20)

    upper_tail_top_errors = tail_pairwise.sort_values(
        "upper_tail_absolute_error",
        ascending=False,
    ).head(20)

    lower_tail_top_errors.to_csv(
        output_dir / "lower_tail_top_pairwise_errors.csv",
        index=False,
    )

    upper_tail_top_errors.to_csv(
        output_dir / "upper_tail_top_pairwise_errors.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 6. Overall summary
    # ------------------------------------------------------------------
    summary = {
        "dataset_id": dataset_id,
        "real_shape": list(real_df.shape),
        "generated_shape": list(generated_df.shape),
        "generated_file": str(generated_path),
        "graph_file": (
            None
            if args.graph is None
            else str(resolve_project_path(args.graph))
        ),
        "marginal": {
            "mean_abs_quantile_error": float(
                marginal_errors["absolute_error"].mean()
            ),
            "median_abs_quantile_error": float(
                marginal_errors["absolute_error"].median()
            ),
            "max_abs_quantile_error": float(
                marginal_errors["absolute_error"].max()
            ),
            "mean_normalized_abs_quantile_error": float(
                marginal_errors["normalized_absolute_error"].mean()
            ),
            "median_normalized_abs_quantile_error": float(
                marginal_errors["normalized_absolute_error"].median()
            ),
            "max_normalized_abs_quantile_error": float(
                marginal_errors["normalized_absolute_error"].max()
            ),
        },
        "kendall": kendall_summary,
        "spearman": spearman_summary,
        "graph": graph_summary,
        "graph_dependence": graph_dependence_summary,
        "tail_dependence": {
            "scale": "empirical_rank_pseudo_observations",
            **tail_summary,
        },
    }

    summary_path = output_dir / "evaluation_summary.json"
    save_json(summary, summary_path)

    print("\nEvaluation complete.")
    print(f"Output directory: {output_dir}")

    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
