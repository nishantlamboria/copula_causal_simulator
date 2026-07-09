"""Run the complete synthetic-validation suite in one command.

This runner combines the already completed pair-family benchmark with four
remaining validation layers: empirical marginals, known three-variable
D-vines and parent-order selection, a complete five-node DAG, and a controlled
violation of the simplifying assumption.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from copula_causal_sim.synthetic.validation_suite import (  # noqa: E402
    KnownDagSpecification,
    PairSpec,
    build_fit_controls,
    run_dvine_validation,
    run_full_dag_validation,
    run_marginal_validation,
    run_non_simplified_validation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all synthetic-validation experiments.")
    parser.add_argument("--config", required=True, help="Path to YAML configuration.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output directory.")
    return parser.parse_args()


def resolve(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    return config


def pair_spec(payload: dict[str, Any]) -> PairSpec:
    return PairSpec(
        family=str(payload["family"]),
        tau=float(payload["tau"]),
        student_df=float(payload.get("student_df", 4.0)),
    )


def summarize_pair_results(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "path": str(path)}
    data = pd.read_csv(path)
    successful = data.loc[data["status"] == "success"].copy()
    if successful.empty:
        return {"available": True, "successful_runs": 0, "path": str(path)}
    boolean = successful["family_correct"]
    if not pd.api.types.is_bool_dtype(boolean):
        boolean = boolean.astype(str).str.lower().map({"true": True, "false": False})
    return {
        "available": True,
        "path": str(path),
        "successful_runs": int(len(successful)),
        "family_recovery_accuracy": float(boolean.mean()),
        "mean_tau_abs_error": float(successful["tau_abs_error"].mean()),
    }


def dataframe_summary(frame: pd.DataFrame, metrics: dict[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {"number_runs": int(len(frame))}
    for output_name, expression in metrics.items():
        column, operation = expression.split(":", maxsplit=1)
        series = frame[column]
        if operation == "mean":
            result[output_name] = float(series.mean())
        elif operation == "median":
            result[output_name] = float(series.median())
        else:
            raise ValueError(f"Unknown summary operation {operation!r}.")
    return result


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    config = load_config(config_path)
    output_directory = resolve(config["output_directory"])
    if output_directory.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory exists: {output_directory}. Use --overwrite."
            )
        shutil.rmtree(output_directory)
    output_directory.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output_directory / "experiment_config.yaml")

    fit_config = config.get("fitting", {})
    controls = build_fit_controls(
        candidate_families=fit_config.get(
            "candidate_families",
            ["indep", "gaussian", "student", "clayton", "gumbel", "frank"],
        ),
        selection_criterion=fit_config.get("selection_criterion", "bic"),
        allow_rotations=bool(fit_config.get("allow_rotations", True)),
    )

    print("[1/4] Empirical marginal and pair-copula validation")
    marginal_config = config["marginal_validation"]
    marginal_results = run_marginal_validation(
        families=marginal_config["families"],
        taus=[float(value) for value in marginal_config["taus"]],
        marginal_pairs=[tuple(value) for value in marginal_config["marginal_pairs"]],
        sample_sizes=[int(value) for value in marginal_config["sample_sizes"]],
        seeds=[int(value) for value in marginal_config["seeds"]],
        controls=controls,
    )
    marginal_results.to_csv(output_directory / "marginal_validation_results.csv", index=False)

    print("[2/4] Known D-vine recovery and parent-order selection")
    dvine_config = config["dvine_validation"]
    dvine_results, order_candidates = run_dvine_validation(
        scenarios=dvine_config["scenarios"],
        sample_sizes=[int(value) for value in dvine_config["sample_sizes"]],
        seeds=[int(value) for value in dvine_config["seeds"]],
        controls=controls,
    )
    dvine_results.to_csv(output_directory / "dvine_validation_results.csv", index=False)
    order_candidates.to_csv(output_directory / "dvine_order_candidates.csv", index=False)

    print("[3/4] Complete known-DAG calibration and regeneration")
    dag_config = config["full_dag_validation"]
    dag_specification = KnownDagSpecification(
        x1_x2=pair_spec(dag_config["specification"]["x1_x2"]),
        x2_x3=pair_spec(dag_config["specification"]["x2_x3"]),
        x1_x3_given_x2=pair_spec(
            dag_config["specification"]["x1_x3_given_x2"]
        ),
        x3_x4=pair_spec(dag_config["specification"]["x3_x4"]),
        x2_x5=pair_spec(dag_config["specification"]["x2_x5"]),
    )
    dag_results = run_full_dag_validation(
        specification=dag_specification,
        sample_sizes=[int(value) for value in dag_config["sample_sizes"]],
        seeds=[int(value) for value in dag_config["seeds"]],
        controls=controls,
    )
    dag_results.to_csv(output_directory / "full_dag_validation_results.csv", index=False)

    print("[4/4] Simplifying-assumption violation and adaptive binning")
    non_simplified_config = config["non_simplified_validation"]
    non_simplified_results, non_simplified_curves = run_non_simplified_validation(
        sample_sizes=[int(value) for value in non_simplified_config["sample_sizes"]],
        seeds=[int(value) for value in non_simplified_config["seeds"]],
        candidate_bins=[int(value) for value in non_simplified_config["candidate_bins"]],
        min_bin_size=int(non_simplified_config["min_bin_size"]),
        controls=controls,
    )
    non_simplified_results.to_csv(
        output_directory / "non_simplified_validation_results.csv", index=False
    )
    non_simplified_curves.to_csv(
        output_directory / "non_simplified_tau_curves.csv", index=False
    )

    pair_results_path = resolve(config["existing_pair_results"])
    if not pair_results_path.exists():
        fallback = PROJECT_ROOT / "pair_family_recovery_full" / "pair_family_recovery_results.csv"
        if fallback.exists():
            pair_results_path = fallback
    summary = {
        "pair_family_validation": summarize_pair_results(pair_results_path),
        "marginal_validation": dataframe_summary(
            marginal_results,
            {
                "family_recovery_accuracy": "family_correct:mean",
                "mean_tau_abs_error": "tau_abs_error:mean",
                "mean_marginal_quantile_error": "mean_marginal_error:mean",
                "max_rank_invariance_error": "max_rank_invariance_error:mean",
            },
        ),
        "dvine_validation": dataframe_summary(
            dvine_results,
            {
                "generating_order_selection_rate": "generating_order_selected:mean",
                "first_tree_family_recovery": "true_order_first_middle_correct:mean",
                "second_first_tree_family_recovery": "true_order_middle_child_correct:mean",
                "conditional_family_recovery": "true_order_conditional_correct:mean",
                "mean_pairwise_kendall_mae": "pairwise_kendall_mae:mean",
                "mean_energy_distance_3d": "energy_distance_3d:mean",
                "mean_test_loglik_gap_from_oracle": "test_loglik_gap_from_oracle:mean",
            },
        ),
        "full_dag_validation": dataframe_summary(
            dag_results,
            {
                "generating_order_selection_rate": "generating_x3_order_selected:mean",
                "one_parent_x2_family_recovery": "x2_family_correct:mean",
                "one_parent_x4_family_recovery": "x4_family_correct:mean",
                "one_parent_x5_family_recovery": "x5_family_correct:mean",
                "mean_pairwise_kendall_mae": "pairwise_kendall_mae:mean",
                "mean_energy_distance_5d": "energy_distance_5d:mean",
                "mean_marginal_quantile_error": "mean_marginal_quantile_error:mean",
            },
        ),
        "non_simplified_validation": dataframe_summary(
            non_simplified_results,
            {
                "mean_selected_bins": "selected_bins:mean",
                "mean_test_loglik_improvement": "binned_test_loglik_improvement:mean",
                "mean_simplified_tau_rmse": "simplified_tau_rmse:mean",
                "mean_binned_tau_rmse": "binned_tau_rmse:mean",
            },
        ),
    }
    with (output_directory / "synthetic_validation_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)

    print("Complete synthetic validation finished.")
    print(json.dumps(summary, indent=2))
    print(f"Outputs: {output_directory}")


if __name__ == "__main__":
    main()
