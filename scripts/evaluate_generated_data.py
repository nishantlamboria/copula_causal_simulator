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
    save_dataframe,
    save_json,
)
from copula_causal_sim.graphs.dag import summarize_dag


def parse_args():
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

    return parser.parse_args()


def load_graph_summary(graph_path: str | None) -> dict | None:
    if graph_path is None:
        return None

    path = Path(graph_path)

    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")

    adjacency = pd.read_csv(path, header=None).to_numpy(dtype=int)
    return summarize_dag(adjacency)


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    dataset_id = config["dataset_id"]

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating generated data for dataset: {dataset_id}")

    print("Loading real calibration data...")
    real_df = load_tabular_dataset(config)

    print("Loading generated data...")
    generated_path = Path(args.generated_x)

    if not generated_path.exists():
        raise FileNotFoundError(f"Generated data file not found: {generated_path}")

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

    marginal_errors_path = output_dir / "marginal_quantile_errors.csv"
    marginal_summary_path = output_dir / "marginal_error_summary.csv"

    marginal_errors.to_csv(marginal_errors_path, index=False)
    marginal_summary.to_csv(marginal_summary_path, index=False)

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
    # 4. Graph metadata
    # ------------------------------------------------------------------
    print("Reading graph metadata...")

    graph_summary = load_graph_summary(args.graph)

    if graph_summary is not None:
        save_json(graph_summary, output_dir / "graph_summary.json")

    # ------------------------------------------------------------------
    # 5. Overall summary
    # ------------------------------------------------------------------
    summary = {
        "dataset_id": dataset_id,
        "real_shape": list(real_df.shape),
        "generated_shape": list(generated_df.shape),
        "generated_file": str(generated_path),
        "graph_file": args.graph,
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
        },
        "kendall": kendall_summary,
        "spearman": spearman_summary,
        "graph": graph_summary,
    }

    summary_path = output_dir / "evaluation_summary.json"
    save_json(summary, summary_path)

    print("\nEvaluation complete.")
    print(f"Output directory: {output_dir}")

    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()