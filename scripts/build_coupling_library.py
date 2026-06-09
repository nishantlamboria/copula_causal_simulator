from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from copula_causal_sim.config import load_config
from copula_causal_sim.data.loaders import (
    load_tabular_dataset,
    make_dataset_summary,
)
from copula_causal_sim.copulas.marginals import (
    make_pseudo_observations,
    EmpiricalMarginalLibrary,
)
from copula_causal_sim.copulas.paircopula_library import (
    build_paircopula_library,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a copula coupling library from a tabular dataset."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to dataset YAML config.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    dataset_id = config["dataset_id"]
    clip_eps = float(config["data"].get("clip_eps", 1e-6))

    artifact_dir = PROJECT_ROOT / config["outputs"]["artifact_dir"]
    table_dir = PROJECT_ROOT / config["outputs"]["table_dir"]
    model_dir = artifact_dir / "paircopula_models"

    artifact_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building coupling library for dataset: {dataset_id}")

    print("Loading data...")
    df = load_tabular_dataset(config)

    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    print("Saving dataset summary...")
    summary = make_dataset_summary(df)
    summary.to_csv(table_dir / "dataset_summary.csv", index=False)

    print("Creating pseudo-observations...")
    u_df = make_pseudo_observations(df, clip_eps=clip_eps)
    u_df.to_csv(artifact_dir / "pseudo_observations.csv", index=False)

    print("Fitting empirical marginals...")
    marginal_library = EmpiricalMarginalLibrary.fit(
        df=df,
        dataset_id=dataset_id,
        clip_eps=clip_eps,
    )
    marginal_library.save(artifact_dir / "marginals.pkl")

    print("Fitting pairwise copula library...")
    pair_library = build_paircopula_library(
        u_df=u_df,
        dataset_id=dataset_id,
        selection_criterion=config["copula"].get("selection_criterion", "bic"),
        allow_rotations=bool(config["copula"].get("allow_rotations", True)),
        num_threads=int(config["copula"].get("num_threads", 1)),
        model_dir=model_dir,
    )

    pair_summary = pair_library.to_dataframe()
    pair_summary.to_csv(table_dir / "paircopula_summary.csv", index=False)
    pair_library.save_pickle(artifact_dir / "coupling_library.pkl")

    print("\nCoupling library build complete.")
    print(f"Dataset: {dataset_id}")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Number of pair copulas: {len(pair_summary)}")
    print(f"Artifacts saved to: {artifact_dir}")
    print(f"Tables saved to: {table_dir}")

    print("\nSelected family counts:")
    print(pair_summary["family"].value_counts())


if __name__ == "__main__":
    main()