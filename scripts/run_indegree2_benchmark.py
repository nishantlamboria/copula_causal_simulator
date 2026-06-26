from __future__ import annotations

from pathlib import Path
import argparse
import json
import subprocess
import sys
import time
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from copula_causal_sim.config import load_config
from copula_causal_sim.data.loaders import load_tabular_dataset


TAIL_METRIC_KEYS = [
    "tail_probability",
    "tail_num_all_pairs",
    "tail_lower_all_real_mean",
    "tail_lower_all_generated_mean",
    "tail_lower_all_mean_abs_error",
    "tail_lower_all_median_abs_error",
    "tail_lower_all_max_abs_error",
    "tail_upper_all_real_mean",
    "tail_upper_all_generated_mean",
    "tail_upper_all_mean_abs_error",
    "tail_upper_all_median_abs_error",
    "tail_upper_all_max_abs_error",
    "tail_num_edge_pairs",
    "tail_lower_edge_real_mean",
    "tail_lower_edge_generated_mean",
    "tail_lower_edge_mean_abs_error",
    "tail_lower_edge_median_abs_error",
    "tail_lower_edge_max_abs_error",
    "tail_upper_edge_real_mean",
    "tail_upper_edge_generated_mean",
    "tail_upper_edge_mean_abs_error",
    "tail_upper_edge_median_abs_error",
    "tail_upper_edge_max_abs_error",
    "tail_num_nonedge_pairs",
    "tail_lower_nonedge_real_mean",
    "tail_lower_nonedge_generated_mean",
    "tail_lower_nonedge_mean_abs_error",
    "tail_lower_nonedge_median_abs_error",
    "tail_lower_nonedge_max_abs_error",
    "tail_upper_nonedge_real_mean",
    "tail_upper_nonedge_generated_mean",
    "tail_upper_nonedge_mean_abs_error",
    "tail_upper_nonedge_median_abs_error",
    "tail_upper_nonedge_max_abs_error",
    "tail_lower_generated_edge_nonedge_gap",
    "tail_upper_generated_edge_nonedge_gap",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run full indegree-2 benchmark: coupling library, conditional vines, "
            "random DAGs, generation, tail-aware evaluation, and runtime measurement."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Dataset config path, e.g. configs/datasets/sachs.yaml",
    )

    parser.add_argument(
        "--num-graphs",
        type=int,
        default=3,
        help="Number of random DAGs to generate.",
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of synthetic samples per graph.",
    )

    parser.add_argument(
        "--edge-prob",
        type=float,
        default=0.30,
        help="Random graph edge probability.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master random seed.",
    )

    parser.add_argument(
        "--tail-probability",
        type=float,
        default=0.05,
        help="Tail probability used by the evaluator. Must satisfy 0 < p < 0.5.",
    )

    parser.add_argument(
        "--rebuild-coupling-library",
        action="store_true",
        help="Force rebuilding empirical marginals and pair-copula library.",
    )

    parser.add_argument(
        "--rebuild-conditional-vines",
        action="store_true",
        help="Force rebuilding conditional D-vine library for indegree 2.",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs/benchmarks",
        help="Root output directory for benchmarks.",
    )

    return parser.parse_args()


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def run_command(command: list[str]) -> float:
    print("\n" + "=" * 80)
    print("Running command:")
    print(" ".join(command))
    print("=" * 80)

    start = time.perf_counter()

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )

    elapsed = time.perf_counter() - start
    print(f"Command runtime: {elapsed:.3f} seconds")

    return elapsed


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(payload: dict | list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def extract_tail_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    tail_summary = summary.get("tail_dependence") or {}

    return {
        key: tail_summary.get(key)
        for key in TAIL_METRIC_KEYS
    }


def safe_rate(n_samples: int, runtime_seconds: float) -> float | None:
    if runtime_seconds <= 0.0:
        return None

    return float(n_samples / runtime_seconds)


def main() -> None:
    args = parse_args()

    if args.num_graphs <= 0:
        raise ValueError("--num-graphs must be positive.")

    if args.n_samples <= 0:
        raise ValueError("--n-samples must be positive.")

    if not 0.0 < args.tail_probability < 0.5:
        raise ValueError("--tail-probability must satisfy 0 < p < 0.5.")

    config_path = resolve_project_path(args.config)
    config = load_config(config_path)

    dataset_id = config["dataset_id"]
    artifact_dir = resolve_project_path(config["outputs"]["artifact_dir"])

    real_df = load_tabular_dataset(config)
    num_nodes = real_df.shape[1]

    benchmark_dir = (
        resolve_project_path(args.output_root)
        / dataset_id
        / "indegree2"
        / f"seed_{args.seed}"
    )

    graphs_dir = benchmark_dir / "graphs"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    print("Benchmark settings:")
    print(f"Dataset: {dataset_id}")
    print(f"Number of variables/nodes: {num_nodes}")
    print(f"Number of graphs: {args.num_graphs}")
    print(f"Synthetic samples per graph: {args.n_samples}")
    print(f"Edge probability: {args.edge_prob}")
    print("Max indegree: 2")
    print(f"Tail probability: {args.tail_probability}")
    print(f"Seed: {args.seed}")
    print(f"Benchmark directory: {benchmark_dir}")

    # ------------------------------------------------------------------
    # 1. Build or reuse coupling library
    # ------------------------------------------------------------------
    coupling_library_exists = (
        (artifact_dir / "marginals.pkl").exists()
        and (artifact_dir / "coupling_library.pkl").exists()
        and (artifact_dir / "paircopula_models").exists()
        and (artifact_dir / "pseudo_observations.csv").exists()
    )

    if args.rebuild_coupling_library or not coupling_library_exists:
        print(
            "\nCoupling library missing or rebuild requested. "
            "Building coupling library..."
        )

        run_command([
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_coupling_library.py"),
            "--config",
            str(config_path),
        ])
    else:
        print("\nCoupling library already exists. Reusing existing library.")

    # ------------------------------------------------------------------
    # 2. Build or reuse conditional D-vine library
    # ------------------------------------------------------------------
    conditional_vine_exists = (
        (artifact_dir / "conditional_vine_library_indegree2.pkl").exists()
        and (artifact_dir / "conditional_vines_indegree2").exists()
    )

    if args.rebuild_conditional_vines or not conditional_vine_exists:
        print(
            "\nConditional D-vine library missing or rebuild requested. "
            "Building it..."
        )

        run_command([
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_conditional_vine_library.py"),
            "--config",
            str(config_path),
            "--parent-set-size",
            "2",
        ])
    else:
        print(
            "\nConditional D-vine library already exists. "
            "Reusing existing library."
        )

    # ------------------------------------------------------------------
    # 3. Generate random DAGs with max indegree 2
    # ------------------------------------------------------------------
    run_command([
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "generate_random_graphs.py"),
        "--num-graphs",
        str(args.num_graphs),
        "--num-nodes",
        str(num_nodes),
        "--edge-prob",
        str(args.edge_prob),
        "--max-indegree",
        "2",
        "--seed",
        str(args.seed),
        "--output-dir",
        str(graphs_dir.relative_to(PROJECT_ROOT)),
    ])

    # ------------------------------------------------------------------
    # 4. Generate data and evaluate each graph
    # ------------------------------------------------------------------
    rows: list[dict[str, Any]] = []

    for graph_idx in range(args.num_graphs):
        graph_id = f"graph_{graph_idx:04d}"
        generation_seed = args.seed + graph_idx

        graph_path = graphs_dir / f"{graph_id}_adjacency.csv"
        graph_output_dir = benchmark_dir / graph_id
        generated_dir = graph_output_dir / "generated"
        evaluation_dir = graph_output_dir / "evaluation"

        if not graph_path.exists():
            raise FileNotFoundError(f"Expected graph file not found: {graph_path}")

        print("\n" + "#" * 80)
        print(f"Processing {graph_id}")
        print("#" * 80)

        generation_runtime_seconds = run_command([
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "generate_from_graph_indegree2.py"),
            "--config",
            str(config_path),
            "--graph",
            str(graph_path),
            "--n-samples",
            str(args.n_samples),
            "--seed",
            str(generation_seed),
            "--output-dir",
            str(generated_dir.relative_to(PROJECT_ROOT)),
        ])

        generated_x_path = generated_dir / "generated_x.csv"
        generated_graph_path = generated_dir / "adjacency.csv"

        if not generated_x_path.exists():
            raise FileNotFoundError(
                f"Generated data file not found after generation: {generated_x_path}"
            )

        if not generated_graph_path.exists():
            raise FileNotFoundError(
                f"Generated adjacency file not found: {generated_graph_path}"
            )

        evaluation_runtime_seconds = run_command([
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluate_generated_data.py"),
            "--config",
            str(config_path),
            "--generated-x",
            str(generated_x_path.relative_to(PROJECT_ROOT)),
            "--graph",
            str(generated_graph_path.relative_to(PROJECT_ROOT)),
            "--output-dir",
            str(evaluation_dir.relative_to(PROJECT_ROOT)),
            "--tail-probability",
            str(args.tail_probability),
        ])

        total_graph_runtime_seconds = (
            generation_runtime_seconds + evaluation_runtime_seconds
        )

        evaluation_summary_path = evaluation_dir / "evaluation_summary.json"

        if not evaluation_summary_path.exists():
            raise FileNotFoundError(
                f"Evaluation summary not found: {evaluation_summary_path}"
            )

        summary = load_json(evaluation_summary_path)
        graph_summary = summary.get("graph") or {}
        graph_dependence = summary.get("graph_dependence") or {}
        kendall_graph = graph_dependence.get("kendall") or {}
        spearman_graph = graph_dependence.get("spearman") or {}

        row: dict[str, Any] = {
            "dataset_id": dataset_id,
            "generator": "indegree_two_conditional_dvine",
            "graph_id": graph_id,
            "seed": generation_seed,
            "n_samples": args.n_samples,
            "num_nodes": graph_summary.get("num_nodes"),
            "num_edges": graph_summary.get("num_edges"),
            "max_indegree": graph_summary.get("max_indegree"),
            "max_outdegree": graph_summary.get("max_outdegree"),

            "marginal_mean_abs_quantile_error": summary["marginal"]["mean_abs_quantile_error"],
            "marginal_median_abs_quantile_error": summary["marginal"]["median_abs_quantile_error"],
            "marginal_max_abs_quantile_error": summary["marginal"]["max_abs_quantile_error"],
            "marginal_mean_normalized_abs_quantile_error": summary["marginal"]["mean_normalized_abs_quantile_error"],
            "marginal_median_normalized_abs_quantile_error": summary["marginal"]["median_normalized_abs_quantile_error"],
            "marginal_max_normalized_abs_quantile_error": summary["marginal"]["max_normalized_abs_quantile_error"],

            "kendall_mean_abs_error": summary["kendall"]["mean_abs_error"],
            "kendall_median_abs_error": summary["kendall"]["median_abs_error"],
            "kendall_max_abs_error": summary["kendall"]["max_abs_error"],
            "spearman_mean_abs_error": summary["spearman"]["mean_abs_error"],
            "spearman_median_abs_error": summary["spearman"]["median_abs_error"],
            "spearman_max_abs_error": summary["spearman"]["max_abs_error"],

            "kendall_num_edge_pairs": kendall_graph.get("num_edge_pairs"),
            "kendall_num_nonedge_pairs": kendall_graph.get("num_nonedge_pairs"),
            "kendall_edge_generated_mean_abs_value": kendall_graph.get("edge_generated_mean_abs_value"),
            "kendall_nonedge_generated_mean_abs_value": kendall_graph.get("nonedge_generated_mean_abs_value"),
            "kendall_generated_edge_nonedge_abs_gap": kendall_graph.get("generated_edge_nonedge_abs_gap"),
            "kendall_edge_mean_abs_error": kendall_graph.get("edge_mean_abs_error"),
            "kendall_nonedge_mean_abs_error": kendall_graph.get("nonedge_mean_abs_error"),

            "spearman_num_edge_pairs": spearman_graph.get("num_edge_pairs"),
            "spearman_num_nonedge_pairs": spearman_graph.get("num_nonedge_pairs"),
            "spearman_edge_generated_mean_abs_value": spearman_graph.get("edge_generated_mean_abs_value"),
            "spearman_nonedge_generated_mean_abs_value": spearman_graph.get("nonedge_generated_mean_abs_value"),
            "spearman_generated_edge_nonedge_abs_gap": spearman_graph.get("generated_edge_nonedge_abs_gap"),
            "spearman_edge_mean_abs_error": spearman_graph.get("edge_mean_abs_error"),
            "spearman_nonedge_mean_abs_error": spearman_graph.get("nonedge_mean_abs_error"),

            "generation_runtime_seconds": generation_runtime_seconds,
            "evaluation_runtime_seconds": evaluation_runtime_seconds,
            "total_graph_runtime_seconds": total_graph_runtime_seconds,
            "generation_samples_per_second": safe_rate(
                args.n_samples,
                generation_runtime_seconds,
            ),
            "end_to_end_samples_per_second": safe_rate(
                args.n_samples,
                total_graph_runtime_seconds,
            ),

            "generated_x_file": str(generated_x_path.relative_to(PROJECT_ROOT)),
            "evaluation_dir": str(evaluation_dir.relative_to(PROJECT_ROOT)),
        }

        row.update(extract_tail_metrics(summary))
        rows.append(row)

    # ------------------------------------------------------------------
    # 5. Save benchmark summary
    # ------------------------------------------------------------------
    summary_df = pd.DataFrame(rows)

    summary_csv_path = benchmark_dir / "benchmark_summary.csv"
    summary_json_path = benchmark_dir / "benchmark_summary.json"

    summary_df.to_csv(summary_csv_path, index=False)
    save_json(rows, summary_json_path)

    print("\n" + "=" * 80)
    print("Indegree-2 benchmark complete.")
    print("=" * 80)
    print(f"Benchmark directory: {benchmark_dir}")
    print(f"Summary CSV: {summary_csv_path}")
    print(f"Summary JSON: {summary_json_path}")

    display_columns = [
        "graph_id",
        "num_edges",
        "marginal_median_normalized_abs_quantile_error",
        "kendall_mean_abs_error",
        "spearman_mean_abs_error",
        "tail_lower_all_mean_abs_error",
        "tail_upper_all_mean_abs_error",
        "generation_runtime_seconds",
        "generation_samples_per_second",
    ]

    existing_display_columns = [
        column
        for column in display_columns
        if column in summary_df.columns
    ]

    print("\nBenchmark summary:")
    print(
        summary_df[existing_display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )


if __name__ == "__main__":
    main()
