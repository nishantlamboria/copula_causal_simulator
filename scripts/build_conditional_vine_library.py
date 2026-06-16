from pathlib import Path
import argparse
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from copula_causal_sim.config import load_config
from copula_causal_sim.data.loaders import load_tabular_dataset
from copula_causal_sim.copulas.marginals import make_pseudo_observations
from copula_causal_sim.copulas.conditional_vine_library import (
    build_conditional_vine_library,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build conditional vine library for bounded-indegree generation."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Dataset YAML config.",
    )

    parser.add_argument(
        "--parent-set-size",
        type=int,
        default=2,
        help="Number of parents in local conditional mechanism. Currently only 2.",
    )

    parser.add_argument(
        "--selection-criterion",
        type=str,
        default=None,
        choices=["aic", "bic", "mbic"],
        help="Override copula selection criterion from config.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for smoke testing. Example: --limit 10",
    )

    parser.add_argument(
        "--no-fixed-variable-order",
        action="store_true",
        help=(
            "Allow automatic vine structure selection instead of requesting "
            "fixed [parent_1, parent_2, child] order."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = PROJECT_ROOT / args.config
    config = load_config(config_path)

    dataset_id = config["dataset_id"]
    clip_eps = float(config["data"].get("clip_eps", 1e-6))

    selection_criterion = (
        args.selection_criterion
        if args.selection_criterion is not None
        else config["copula"].get("selection_criterion", "bic")
    )

    artifact_dir = PROJECT_ROOT / config["outputs"]["artifact_dir"]
    table_dir = PROJECT_ROOT / config["outputs"]["table_dir"]

    conditional_model_dir = (
        artifact_dir
        / f"conditional_vines_indegree{args.parent_set_size}"
    )

    conditional_library_path = (
        artifact_dir
        / f"conditional_vine_library_indegree{args.parent_set_size}.pkl"
    )

    conditional_summary_path = (
        table_dir
        / f"conditional_vine_summary_indegree{args.parent_set_size}.csv"
    )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    conditional_model_dir.mkdir(parents=True, exist_ok=True)

    print("Building conditional vine library")
    print(f"Dataset: {dataset_id}")
    print(f"Config: {config_path}")
    print(f"Parent set size: {args.parent_set_size}")
    print(f"Selection criterion: {selection_criterion}")
    print(f"Model directory: {conditional_model_dir}")

    pseudo_path = artifact_dir / "pseudo_observations.csv"

    if pseudo_path.exists():
        print(f"Loading existing pseudo-observations: {pseudo_path}")
        u_df = pd.read_csv(pseudo_path)
    else:
        print("Pseudo-observations not found. Loading data and creating them...")
        df = load_tabular_dataset(config)
        u_df = make_pseudo_observations(df, clip_eps=clip_eps)
        u_df.to_csv(pseudo_path, index=False)
        print(f"Saved pseudo-observations to: {pseudo_path}")

    print(f"Pseudo-observation shape: {u_df.shape}")
    print(f"Columns: {list(u_df.columns)}")

    fixed_variable_order = not args.no_fixed_variable_order

    library = build_conditional_vine_library(
        u_df=u_df,
        dataset_id=dataset_id,
        parent_set_size=args.parent_set_size,
        selection_criterion=selection_criterion,
        allow_rotations=bool(config["copula"].get("allow_rotations", True)),
        num_threads=int(config["copula"].get("num_threads", 1)),
        model_dir=conditional_model_dir,
        fixed_variable_order=fixed_variable_order,
        limit=args.limit,
    )

    summary_df = library.to_dataframe()

    summary_df.to_csv(conditional_summary_path, index=False)
    library.save_pickle(conditional_library_path)

    n_ok = int((summary_df["status"] == "ok").sum())
    n_failed = int((summary_df["status"] == "failed").sum())

    print("\nConditional vine library build complete.")
    print(f"Total attempted models: {len(summary_df)}")
    print(f"Successful fits: {n_ok}")
    print(f"Failed fits: {n_failed}")
    print(f"Summary saved to: {conditional_summary_path}")
    print(f"Library metadata saved to: {conditional_library_path}")
    print(f"Individual vine models saved in: {conditional_model_dir}")

    if n_failed > 0:
        print("\nFailed model examples:")
        failed_cols = ["child", "parent_1", "parent_2", "error"]
        print(summary_df.loc[summary_df["status"] == "failed", failed_cols].head(10))


if __name__ == "__main__":
    main()