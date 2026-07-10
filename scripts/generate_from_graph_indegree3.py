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
from copula_causal_sim.copulas.marginals import EmpiricalMarginalLibrary
from copula_causal_sim.generators.indegree_one import load_paircopula_metadata
from copula_causal_sim.generators.indegree_two import (
    load_conditional_vine_metadata,
)
from copula_causal_sim.generators.indegree_three import (
    IndegreeThreeGenerator,
    load_conditional_vine_metadata_indegree3,
)
from copula_causal_sim.graphs.dag import summarize_dag, validate_dag


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic data from a DAG with max indegree <= 3."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Dataset config used to locate the fitted libraries.",
    )

    parser.add_argument(
        "--graph",
        type=str,
        required=True,
        help="Path to adjacency matrix CSV. Convention: adj[i,j]=1 means i -> j.",
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of synthetic samples to generate.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/generated_indegree3",
        help="Output directory.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    dataset_id = config["dataset_id"]

    artifact_dir = PROJECT_ROOT / config["outputs"]["artifact_dir"]
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    marginal_path = artifact_dir / "marginals.pkl"
    paircopula_path = artifact_dir / "coupling_library.pkl"
    conditional_vine_path = artifact_dir / "conditional_vine_library_indegree2.pkl"
    conditional_vine_path_indegree3 = artifact_dir / "conditional_vine_library_indegree3.pkl"

    if not marginal_path.exists():
        raise FileNotFoundError(
            f"Marginal library not found: {marginal_path}\n"
            "Run scripts/build_coupling_library.py first."
        )

    if not paircopula_path.exists():
        raise FileNotFoundError(
            f"Pair-copula library not found: {paircopula_path}\n"
            "Run scripts/build_coupling_library.py first."
        )

    if not conditional_vine_path.exists():
        raise FileNotFoundError(
            f"Indegree-two conditional vine library not found: {conditional_vine_path}\n"
            "Build it with --parent-set-size 2 first."
        )

    if not conditional_vine_path_indegree3.exists():
        raise FileNotFoundError(
            f"Indegree-three conditional vine library not found: "
            f"{conditional_vine_path_indegree3}\n"
            "Build it with --parent-set-size 3 first."
        )

    print(f"Loading libraries for dataset: {dataset_id}")

    marginal_library = EmpiricalMarginalLibrary.load(marginal_path)
    paircopula_metadata = load_paircopula_metadata(paircopula_path)
    conditional_vine_metadata = load_conditional_vine_metadata(
        conditional_vine_path
    )
    conditional_vine_metadata_indegree3 = (
        load_conditional_vine_metadata_indegree3(
            conditional_vine_path_indegree3
        )
    )

    print(f"Loading graph: {args.graph}")
    adjacency = pd.read_csv(args.graph, header=None).to_numpy(dtype=int)
    validate_dag(adjacency, max_allowed_indegree=3)

    generator = IndegreeThreeGenerator(
        marginal_library=marginal_library,
        paircopula_metadata=paircopula_metadata,
        conditional_vine_metadata_indegree2=conditional_vine_metadata,
        conditional_vine_metadata_indegree3=(
            conditional_vine_metadata_indegree3
        ),
        artifact_dir=artifact_dir,
        clip_eps=float(config["data"].get("clip_eps", 1e-6)),
    )

    print("Generating synthetic data...")
    result = generator.generate(
        adjacency=adjacency,
        n_samples=args.n_samples,
        seed=args.seed,
    )

    u_path = output_dir / "generated_u.csv"
    x_path = output_dir / "generated_x.csv"
    graph_path = output_dir / "adjacency.csv"
    metadata_path = output_dir / "generation_metadata.json"

    result.u.to_csv(u_path, index=False)
    result.x.to_csv(x_path, index=False)
    pd.DataFrame(adjacency).to_csv(graph_path, index=False, header=False)

    metadata = {
        "dataset_id": dataset_id,
        "n_samples": args.n_samples,
        "seed": args.seed,
        "columns": result.columns,
        "graph_summary": summarize_dag(adjacency),
        "generator": "indegree_three_conditional_vine",
        "max_supported_indegree": 3,
        "u_file": str(u_path),
        "x_file": str(x_path),
        "adjacency_file": str(graph_path),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nGeneration complete.")
    print(f"Generated U saved to: {u_path}")
    print(f"Generated X saved to: {x_path}")
    print(f"Metadata saved to: {metadata_path}")
    print("\nSynthetic data preview:")
    print(result.x.head())


if __name__ == "__main__":
    main()