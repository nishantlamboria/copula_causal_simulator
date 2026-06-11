from pathlib import Path
import argparse
import json
import subprocess
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from copula_causal_sim.config import load_config
from copula_causal_sim.data.loaders import load_tabular_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run full indegree-1 benchmark: library, graphs, generation, evaluation."
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
        default=0.25,
        help="Random graph edge probability.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master random seed.",
    )

    parser.add_argument(
        "--rebuild-library",
        action="store_true",
        help="Force rebuilding the coupling library before benchmarking.",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs/benchmarks",
        help="Root output directory for benchmarks.",
    )

    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print("\n" + "=" * 80)
    print("Running command:")
    print(" ".join(command))
    print("=" * 80)

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(payload: dict | list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()

    config_path = PROJECT_ROOT / args.config
    config = load_config(config_path)

    dataset_id = config["dataset_id"]
    artifact_dir = PROJECT_ROOT / config["outputs"]["artifact_dir"]

    # Infer number of nodes from the dataset.
    real_df = load_tabular_dataset(config)
    num_nodes = real_df.shape[1]

    benchmark_dir = (
        PROJECT_ROOT
        / args.output_root
        / dataset_id
        / "indegree1"
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
    print(f"Seed: {args.seed}")
    print(f"Benchmark directory: {benchmark_dir}")

    # ------------------------------------------------------------------
    # 1. Build or reuse coupling library
    # ------------------------------------------------------------------
    library_exists = (
        (artifact_dir / "marginals.pkl").exists()
        and (artifact_dir / "coupling_library.pkl").exists()
        and (artifact_dir / "paircopula_models").exists()
    )

    if args.rebuild_library or not library_exists:
        print("\nCoupling library missing or rebuild requested. Building library...")

        run_command([
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_coupling_library.py"),
            "--config",
            str(config_path),
        ])
    else:
        print("\nCoupling library already exists. Reusing existing library.")

    # ------------------------------------------------------------------
    # 2. Generate random DAGs
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
        "1",
        "--seed",
        str(args.seed),
        "--output-dir",
        str(graphs_dir.relative_to(PROJECT_ROOT)),
    ])

    # ------------------------------------------------------------------
    # 3. Generate data and evaluate each graph
    # ------------------------------------------------------------------
    rows = []

    for graph_idx in range(args.num_graphs):
        graph_id = f"graph_{graph_idx:04d}"

        graph_path = graphs_dir / f"{graph_id}_adjacency.csv"
        graph_output_dir = benchmark_dir / graph_id
        generated_dir = graph_output_dir / "generated"
        evaluation_dir = graph_output_dir / "evaluation"

        if not graph_path.exists():
            raise FileNotFoundError(f"Expected graph file not found: {graph_path}")

        print("\n" + "#" * 80)
        print(f"Processing {graph_id}")
        print("#" * 80)

        # Generate synthetic data from graph.
        run_command([
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "generate_from_graph_indegree1.py"),
            "--config",
            str(config_path),
            "--graph",
            str(graph_path),
            "--n-samples",
            str(args.n_samples),
            "--seed",
            str(args.seed + graph_idx),
            "--output-dir",
            str(generated_dir.relative_to(PROJECT_ROOT)),
        ])

        generated_x_path = generated_dir / "generated_x.csv"
        generated_graph_path = generated_dir / "adjacency.csv"

        # Evaluate generated data.
        run_command([
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
        ])

        evaluation_summary_path = evaluation_dir / "evaluation_summary.json"

        if not evaluation_summary_path.exists():
            raise FileNotFoundError(
                f"Evaluation summary not found: {evaluation_summary_path}"
            )

        summary = load_json(evaluation_summary_path)
        graph_summary = summary.get("graph", {})
        graph_dependence = summary.get("graph_dependence") or {}
        kendall_graph = graph_dependence.get("kendall") or {}
        spearman_graph = graph_dependence.get("spearman") or {}

        row = {
            "dataset_id": dataset_id,
            "graph_id": graph_id,
            "seed": args.seed + graph_idx,
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

            "generated_x_file": str(generated_x_path.relative_to(PROJECT_ROOT)),
            "evaluation_dir": str(evaluation_dir.relative_to(PROJECT_ROOT)),
        }

        rows.append(row)

    # ------------------------------------------------------------------
    # 4. Save benchmark summary
    # ------------------------------------------------------------------
    summary_df = pd.DataFrame(rows)

    summary_csv_path = benchmark_dir / "benchmark_summary.csv"
    summary_json_path = benchmark_dir / "benchmark_summary.json"

    summary_df.to_csv(summary_csv_path, index=False)
    save_json(rows, summary_json_path)

    print("\n" + "=" * 80)
    print("Indegree-1 benchmark complete.")
    print("=" * 80)
    print(f"Benchmark directory: {benchmark_dir}")
    print(f"Summary CSV: {summary_csv_path}")
    print(f"Summary JSON: {summary_json_path}")

    print("\nBenchmark summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()