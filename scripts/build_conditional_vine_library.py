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
        description=(
            "Build flexible-order conditional D-vine library for "
            "bounded-indegree generation."
        )
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
        help="Override pair-copula family selection criterion from config.",
    )

    parser.add_argument(
        "--order-strategy",
        type=str,
        default=None,
        choices=["heldout_conditional_loglik", "bic", "fixed"],
        help=(
            "Parent-order selection strategy. Defaults to "
            "copula.vine_order_strategy in the dataset config."
        ),
    )

    parser.add_argument(
        "--order-validation-fraction",
        type=float,
        default=None,
        help="Validation fraction for held-out conditional log-likelihood.",
    )

    parser.add_argument(
        "--order-selection-seed",
        type=int,
        default=None,
        help="Random seed for the deterministic train/validation split.",
    )

    parser.add_argument(
        "--minimum-validation-rows",
        type=int,
        default=None,
        help=(
            "Minimum validation rows required for held-out selection. "
            "Smaller datasets fall back to BIC."
        ),
    )

    parser.add_argument(
        "--conditional-model-strategy",
        type=str,
        default=None,
        choices=["simplified", "adaptive_quantile_bins"],
        help=(
            "Conditional-copula strategy. Defaults to "
            "copula.conditional_model_strategy in the dataset config."
        ),
    )

    parser.add_argument(
        "--conditional-candidate-bin-counts",
        type=int,
        nargs="+",
        default=None,
        help="Candidate equal-frequency bin counts, for example: 1 2 3 4 5 6.",
    )

    parser.add_argument(
        "--conditional-minimum-bin-size",
        type=int,
        default=None,
        help="Minimum number of training observations in every quantile bin.",
    )

    parser.add_argument(
        "--conditional-validation-fraction",
        type=float,
        default=None,
        help="Validation fraction used to select the number of bins.",
    )

    parser.add_argument(
        "--conditional-selection-seed",
        type=int,
        default=None,
        help="Seed for conditional-model train/validation splitting.",
    )

    parser.add_argument(
        "--conditional-minimum-score-improvement",
        type=float,
        default=None,
        help=(
            "Minimum mean held-out log-likelihood improvement required "
            "before replacing the simplified model with multiple bins."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for smoke testing. Example: --limit 10",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = PROJECT_ROOT / args.config
    config = load_config(config_path)

    dataset_id = config["dataset_id"]
    clip_eps = float(config["data"].get("clip_eps", 1e-6))
    copula_config = config.get("copula", {})

    selection_criterion = (
        args.selection_criterion
        if args.selection_criterion is not None
        else copula_config.get("selection_criterion", "bic")
    )

    order_strategy = (
        args.order_strategy
        if args.order_strategy is not None
        else copula_config.get(
            "vine_order_strategy",
            "heldout_conditional_loglik",
        )
    )

    order_validation_fraction = (
        args.order_validation_fraction
        if args.order_validation_fraction is not None
        else float(
            copula_config.get(
                "vine_order_validation_fraction",
                0.20,
            )
        )
    )

    order_selection_seed = (
        args.order_selection_seed
        if args.order_selection_seed is not None
        else int(
            copula_config.get(
                "vine_order_selection_seed",
                42,
            )
        )
    )

    minimum_validation_rows = (
        args.minimum_validation_rows
        if args.minimum_validation_rows is not None
        else int(
            copula_config.get(
                "vine_order_minimum_validation_rows",
                50,
            )
        )
    )

    conditional_model_strategy = (
        args.conditional_model_strategy
        if args.conditional_model_strategy is not None
        else copula_config.get(
            "conditional_model_strategy",
            "adaptive_quantile_bins",
        )
    )

    conditional_candidate_bin_counts = (
        args.conditional_candidate_bin_counts
        if args.conditional_candidate_bin_counts is not None
        else list(
            copula_config.get(
                "conditional_candidate_bin_counts",
                [1, 2, 3, 4, 5, 6],
            )
        )
    )

    conditional_minimum_bin_size = (
        args.conditional_minimum_bin_size
        if args.conditional_minimum_bin_size is not None
        else int(copula_config.get("conditional_minimum_bin_size", 75))
    )

    conditional_validation_fraction = (
        args.conditional_validation_fraction
        if args.conditional_validation_fraction is not None
        else float(copula_config.get("conditional_validation_fraction", 0.20))
    )

    conditional_selection_seed = (
        args.conditional_selection_seed
        if args.conditional_selection_seed is not None
        else int(copula_config.get("conditional_selection_seed", 42))
    )

    conditional_minimum_score_improvement = (
        args.conditional_minimum_score_improvement
        if args.conditional_minimum_score_improvement is not None
        else float(
            copula_config.get(
                "conditional_minimum_score_improvement",
                0.005,
            )
        )
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

    print("Building flexible-order conditional vine library")
    print(f"Dataset: {dataset_id}")
    print(f"Config: {config_path}")
    print(f"Parent set size: {args.parent_set_size}")
    print(f"Pair-family selection criterion: {selection_criterion}")
    print(f"Order strategy: {order_strategy}")
    print(f"Order validation fraction: {order_validation_fraction}")
    print(f"Order selection seed: {order_selection_seed}")
    print(f"Minimum validation rows: {minimum_validation_rows}")
    print(f"Conditional model strategy: {conditional_model_strategy}")
    print(f"Conditional candidate bins: {conditional_candidate_bin_counts}")
    print(f"Conditional minimum bin size: {conditional_minimum_bin_size}")
    print(f"Conditional validation fraction: {conditional_validation_fraction}")
    print(f"Conditional selection seed: {conditional_selection_seed}")
    print(
        "Conditional minimum score improvement: "
        f"{conditional_minimum_score_improvement}"
    )
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

    library = build_conditional_vine_library(
        u_df=u_df,
        dataset_id=dataset_id,
        parent_set_size=args.parent_set_size,
        selection_criterion=selection_criterion,
        allow_rotations=bool(copula_config.get("allow_rotations", True)),
        num_threads=int(copula_config.get("num_threads", 1)),
        model_dir=conditional_model_dir,
        order_strategy=order_strategy,
        order_validation_fraction=order_validation_fraction,
        order_selection_seed=order_selection_seed,
        minimum_validation_rows=minimum_validation_rows,
        conditional_model_strategy=conditional_model_strategy,
        conditional_candidate_bin_counts=conditional_candidate_bin_counts,
        conditional_minimum_bin_size=conditional_minimum_bin_size,
        conditional_validation_fraction=conditional_validation_fraction,
        conditional_selection_seed=conditional_selection_seed,
        conditional_minimum_score_improvement=(
            conditional_minimum_score_improvement
        ),
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

    if n_ok > 0:
        display_columns = [
            "child",
            "parent_set",
            "selected_order",
            "order_selection_method",
            "selected_order_score",
            "alternative_order_score",
            "order_score_margin",
            "conditional_model_type",
            "selected_number_bins",
            "simplified_validation_score",
            "selected_validation_score",
            "score_improvement_over_simplified",
        ]
        print("\nSelected-order examples:")
        print(summary_df.loc[summary_df["status"] == "ok", display_columns].head(10))

    if n_failed > 0:
        print("\nFailed model examples:")
        failed_cols = ["child", "parent_set", "error"]
        print(summary_df.loc[summary_df["status"] == "failed", failed_cols].head(10))


if __name__ == "__main__":
    main()
