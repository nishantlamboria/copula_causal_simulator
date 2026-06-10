from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from copula_causal_sim.graphs.dag import summarize_dag
from copula_causal_sim.graphs.random_graphs import sample_random_dags


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate random DAGs with bounded indegree."
    )

    parser.add_argument("--num-graphs", type=int, default=5)
    parser.add_argument("--num-nodes", type=int, default=10)
    parser.add_argument("--edge-prob", type=float, default=0.25)
    parser.add_argument("--max-indegree", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/graphs/random_dags",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    graphs = sample_random_dags(
        num_graphs=args.num_graphs,
        num_nodes=args.num_nodes,
        edge_prob=args.edge_prob,
        max_indegree=args.max_indegree,
        seed=args.seed,
    )

    metadata = []

    for idx, adj in enumerate(graphs):
        graph_id = f"graph_{idx:04d}"

        adj_path = output_dir / f"{graph_id}_adjacency.csv"
        meta_path = output_dir / f"{graph_id}_metadata.json"

        pd.DataFrame(adj).to_csv(adj_path, index=False, header=False)

        summary = summarize_dag(adj)
        summary["graph_id"] = graph_id
        summary["adjacency_file"] = str(adj_path)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        metadata.append(summary)

    metadata_path = output_dir / "metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Random DAG generation complete.")
    print(f"Number of graphs: {len(graphs)}")
    print(f"Number of nodes: {args.num_nodes}")
    print(f"Max indegree: {args.max_indegree}")
    print(f"Output directory: {output_dir}")

    edge_counts = [item["num_edges"] for item in metadata]
    max_indegrees = [item["max_indegree"] for item in metadata]

    print(f"Edge counts: {edge_counts}")
    print(f"Observed max indegrees: {max_indegrees}")


if __name__ == "__main__":
    main()