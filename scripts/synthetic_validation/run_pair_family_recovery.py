"""Run synthetic pair-copula family-recovery experiments.

The runner:

1. reads explicit ground-truth scenarios from YAML;
2. generates samples from known pair-copula models;
3. splits each sample into training and test data;
4. selects a candidate copula using the configured criterion;
5. records family, rotation, Kendall-tau, parameter, and likelihood recovery;
6. writes detailed and aggregated CSV files.

Run from the project root:

    python scripts/synthetic_validation/run_pair_family_recovery.py ^
        --config configs/synthetic_validation/pair_family_recovery_intermediate.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyvinecopulib as pv
import yaml
from scipy.stats import kendalltau


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIRECTORY = PROJECT_ROOT / "src"

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from copula_causal_sim.synthetic.pair_copula import (  # noqa: E402
    normalize_family_name,
    simulate_pair_copula,
)


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run pair-copula family and parameter recovery experiments."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to the YAML experiment configuration.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing output directory before running.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve a path relative to the repository root."""
    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file and verify its top-level type."""
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file does not exist: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "The YAML configuration must contain a mapping at the "
            "top level."
        )

    return config


def validate_config(config: dict[str, Any]) -> None:
    """Validate the scenario-based experiment configuration."""
    required_sections = {
        "experiment",
        "data_generation",
        "fitting",
        "evaluation",
        "output",
    }

    missing_sections = required_sections.difference(config)

    if missing_sections:
        raise ValueError(
            "Missing configuration sections: "
            + ", ".join(sorted(missing_sections))
        )

    generation = config["data_generation"]
    fitting = config["fitting"]
    evaluation = config["evaluation"]
    output = config["output"]

    if not isinstance(generation, dict):
        raise TypeError("data_generation must be a mapping.")

    if not isinstance(fitting, dict):
        raise TypeError("fitting must be a mapping.")

    if not isinstance(evaluation, dict):
        raise TypeError("evaluation must be a mapping.")

    if not isinstance(output, dict):
        raise TypeError("output must be a mapping.")

    scenarios = generation.get("scenarios")

    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError(
            "data_generation.scenarios must be a non-empty list."
        )

    for scenario_index, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict):
            raise TypeError(
                f"Scenario {scenario_index} must be a mapping."
            )

        if "family" not in scenario:
            raise ValueError(
                f"Scenario {scenario_index} is missing family."
            )

        family = normalize_family_name(
            str(scenario["family"])
        )

        taus = scenario.get("kendall_taus")

        if not isinstance(taus, list) or not taus:
            raise ValueError(
                f"Scenario {scenario_index} must contain a non-empty "
                "kendall_taus list."
            )

        for tau_value in taus:
            if not isinstance(tau_value, (int, float)):
                raise TypeError(
                    f"Scenario {scenario_index} contains a non-numeric "
                    "Kendall tau."
                )

            tau = float(tau_value)

            if not np.isfinite(tau):
                raise ValueError(
                    f"Scenario {scenario_index} contains a non-finite "
                    "Kendall tau."
                )

            if not -1.0 < tau < 1.0:
                raise ValueError(
                    f"Scenario {scenario_index} has tau={tau}, but tau "
                    "must lie strictly between -1 and 1."
                )

            if family == "indep" and not np.isclose(
                tau,
                0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    "The independence scenario must use tau=0."
                )

            if family in {"clayton", "gumbel"} and tau <= 0.0:
                raise ValueError(
                    f"Unrotated {family} scenarios require positive tau."
                )

        if family == "student":
            student_dfs = scenario.get("student_dfs")

            if not isinstance(student_dfs, list) or not student_dfs:
                raise ValueError(
                    f"Student scenario {scenario_index} must contain "
                    "a non-empty student_dfs list."
                )

            for df_value in student_dfs:
                if not isinstance(df_value, (int, float)):
                    raise TypeError(
                        "Student degrees of freedom must be numeric."
                    )

                degrees_of_freedom = float(df_value)

                if (
                    not np.isfinite(degrees_of_freedom)
                    or degrees_of_freedom <= 2.0
                ):
                    raise ValueError(
                        "Student degrees of freedom must be finite and "
                        "greater than 2."
                    )

        elif "student_dfs" in scenario:
            raise ValueError(
                f"Scenario {scenario_index} supplies student_dfs for "
                f"the non-Student family {family!r}."
            )

    sample_sizes = generation.get("sample_sizes")

    if not isinstance(sample_sizes, list) or not sample_sizes:
        raise ValueError(
            "data_generation.sample_sizes must be a non-empty list."
        )

    for sample_size in sample_sizes:
        if not isinstance(sample_size, int) or sample_size < 4:
            raise ValueError(
                "Every sample size must be an integer of at least 4."
            )

    seeds = generation.get("seeds")

    if not isinstance(seeds, list) or not seeds:
        raise ValueError(
            "data_generation.seeds must be a non-empty list."
        )

    for seed in seeds:
        if not isinstance(seed, int) or seed < 0:
            raise ValueError(
                "Every seed must be a non-negative integer."
            )

    marginal_mode = str(
        generation.get("marginal_mode", "uniform")
    ).lower()

    if marginal_mode != "uniform":
        raise ValueError(
            "This runner currently supports only "
            "data_generation.marginal_mode: uniform."
        )

    candidate_families = fitting.get("candidate_families")

    if (
        not isinstance(candidate_families, list)
        or not candidate_families
    ):
        raise ValueError(
            "fitting.candidate_families must be a non-empty list."
        )

    for family in candidate_families:
        normalize_family_name(str(family))

    criterion = str(
        fitting.get("selection_criterion", "bic")
    ).lower()

    allowed_criteria = {
        "loglik",
        "aic",
        "bic",
        "mbic",
    }

    if criterion not in allowed_criteria:
        raise ValueError(
            f"Unsupported selection criterion {criterion!r}. "
            f"Expected one of {sorted(allowed_criteria)}."
        )

    train_fraction = float(
        evaluation.get("train_fraction", 0.8)
    )

    if not 0.0 < train_fraction < 1.0:
        raise ValueError(
            "evaluation.train_fraction must lie strictly between "
            "0 and 1."
        )

    if "directory" not in output:
        raise ValueError(
            "output.directory must be provided."
        )


def format_float_for_identifier(value: float) -> str:
    """Format a floating-point value for use in directory names."""
    return (
        f"{value:.4f}"
        .rstrip("0")
        .rstrip(".")
        .replace("-", "minus")
        .replace(".", "p")
    )


def make_scenario_id(
    family: str,
    tau: float,
    student_df: float | None,
) -> str:
    """Create a unique identifier for one generating scenario."""
    identifier = (
        f"family_{family}"
        f"__tau_{format_float_for_identifier(tau)}"
    )

    if family == "student":
        if student_df is None:
            raise ValueError(
                "Student scenarios require student_df."
            )

        identifier += (
            f"__df_{format_float_for_identifier(student_df)}"
        )

    return identifier


def expand_scenarios(
    generation_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Expand YAML scenarios into individual ground-truth models."""
    expanded: list[dict[str, Any]] = []

    for scenario in generation_config["scenarios"]:
        family = normalize_family_name(
            str(scenario["family"])
        )

        taus = [
            float(value)
            for value in scenario["kendall_taus"]
        ]

        if family == "student":
            student_dfs: list[float | None] = [
                float(value)
                for value in scenario["student_dfs"]
            ]
        else:
            student_dfs = [None]

        for tau in taus:
            for student_df in student_dfs:
                expanded.append(
                    {
                        "scenario_id": make_scenario_id(
                            family=family,
                            tau=tau,
                            student_df=student_df,
                        ),
                        "family": family,
                        "tau": tau,
                        "student_df": student_df,
                    }
                )

    return expanded


# ---------------------------------------------------------------------
# pyvinecopulib compatibility helpers
# ---------------------------------------------------------------------


def family_enum(family_name: str):
    """Map a configured family name to a pyvinecopulib enum."""
    canonical = normalize_family_name(family_name)

    if hasattr(pv, "BicopFamily"):
        try:
            return getattr(
                pv.BicopFamily,
                canonical,
            )
        except AttributeError as error:
            raise ValueError(
                f"The installed pyvinecopulib version does not expose "
                f"family {canonical!r}."
            ) from error

    try:
        return getattr(pv, canonical)
    except AttributeError as error:
        raise ValueError(
            f"The installed pyvinecopulib version does not expose "
            f"family {canonical!r}."
        ) from error


def family_name_from_model(model: pv.Bicop) -> str:
    """Extract a canonical family name from a fitted model."""
    family = model.family

    enum_name = getattr(family, "name", None)

    if isinstance(enum_name, str):
        return normalize_family_name(enum_name)

    text = str(family).strip().lower()

    if "." in text:
        text = text.split(".")[-1]

    return normalize_family_name(text)


def build_fit_controls(
    candidate_families: list[str],
    selection_criterion: str,
    allow_rotations: bool,
) -> pv.FitControlsBicop:
    """Construct pair-copula fitting controls."""
    family_set = [
        family_enum(family)
        for family in candidate_families
    ]

    arguments = {
        "family_set": family_set,
        "selection_criterion": selection_criterion,
        "parametric_method": "mle",
        "preselect_families": False,
        "num_threads": 1,
    }

    try:
        return pv.FitControlsBicop(
            **arguments,
            allow_rotations=allow_rotations,
        )
    except TypeError:
        controls = pv.FitControlsBicop(**arguments)

        if hasattr(controls, "allow_rotations"):
            try:
                controls.allow_rotations = allow_rotations
            except (AttributeError, TypeError):
                pass

        return controls


def fit_pair_copula(
    train_data: np.ndarray,
    controls: pv.FitControlsBicop,
) -> pv.Bicop:
    """Fit and select a pair copula."""
    data = np.asfortranarray(
        train_data,
        dtype=float,
    )

    if hasattr(pv.Bicop, "from_data"):
        return pv.Bicop.from_data(
            data=data,
            controls=controls,
        )

    model = pv.Bicop()
    model.select(
        data=data,
        controls=controls,
    )

    return model


# ---------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------


def split_train_test(
    data: np.ndarray,
    train_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a reproducible shuffled train/test split."""
    sample_size = data.shape[0]

    train_size = int(
        np.floor(train_fraction * sample_size)
    )

    train_size = max(
        2,
        min(train_size, sample_size - 2),
    )

    rng = np.random.default_rng(
        seed + 1_000_003
    )

    indices = rng.permutation(sample_size)

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_data = np.asfortranarray(
        data[train_indices],
        dtype=float,
    )

    test_data = np.asfortranarray(
        data[test_indices],
        dtype=float,
    )

    return train_data, test_data


def empirical_kendall_tau(data: np.ndarray) -> float:
    """Calculate empirical Kendall's tau."""
    # scipy.stats.kendalltau may return a result object or a tuple depending
    # on versions/type checkers. Unpack to be robust to both.
    kt = kendalltau(
        data[:, 0],
        data[:, 1],
    )

    # kt can be (statistic, pvalue) or an object with .statistic
    # Prefer attribute access, fall back to tuple indexing
    tau_val = getattr(kt, "statistic")
    # if tau_val is None and isinstance(kt, tuple):
    #     tau_val = kt[0]

    tau = float(tau_val)

    if not np.isfinite(tau):
        raise RuntimeError(
            "Empirical Kendall's tau is not finite."
        )

    return tau


def flatten_parameters(model: pv.Bicop) -> list[float]:
    """Return model parameters as a flat Python list."""
    parameters = np.asarray(
        model.parameters,
        dtype=float,
    ).reshape(-1)

    return [
        float(value)
        for value in parameters
    ]


def model_parameter_count(model: pv.Bicop) -> int:
    """Return the model's number of estimated parameters."""
    value = model.npars

    # if callable(value):
    #     value = value()

    return int(value)


def safe_loglik(
    model: pv.Bicop,
    data: np.ndarray,
) -> float:
    """Evaluate a finite copula log-likelihood."""
    value = float(
        model.loglik(
            np.asfortranarray(
                data,
                dtype=float,
            )
        )
    )

    if not np.isfinite(value):
        raise RuntimeError(
            "Model log-likelihood is not finite."
        )

    return value


def safe_bic(
    model: pv.Bicop,
    data: np.ndarray,
) -> float:
    """Evaluate a finite BIC value."""
    value = float(
        model.bic(
            np.asfortranarray(
                data,
                dtype=float,
            )
        )
    )

    if not np.isfinite(value):
        raise RuntimeError(
            "Model BIC is not finite."
        )

    return value


def extract_student_df(
    family: str,
    parameters: list[float],
) -> float:
    """Extract Student-t degrees of freedom when available."""
    if family != "student" or len(parameters) < 2:
        return float("nan")

    return float(parameters[1])


# ---------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------


def save_json(
    payload: dict[str, Any],
    path: Path,
) -> None:
    """Save a JSON-serializable dictionary."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            indent=2,
        )


def save_model(
    model: pv.Bicop,
    path: Path,
) -> None:
    """Save a fitted pyvinecopulib model."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if hasattr(model, "to_file"):
        model.to_file(str(path))
        return

    path.write_text(
        model.to_json(),
        encoding="utf-8",
    )


def make_run_id(
    scenario_id: str,
    sample_size: int,
    seed: int,
) -> str:
    """Create a unique identifier for one experimental run."""
    return (
        f"{scenario_id}"
        f"__n_{sample_size}"
        f"__seed_{seed}"
    )


# ---------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------


def run_single_experiment(
    *,
    scenario_id: str,
    true_family: str,
    true_tau: float,
    true_student_df: float | None,
    sample_size: int,
    seed: int,
    train_fraction: float,
    controls: pv.FitControlsBicop,
    compute_heldout_loglik: bool,
    save_samples: bool,
    runs_directory: Path,
) -> dict[str, Any]:
    """Run one known-model generation and recovery experiment."""
    simulation = simulate_pair_copula(
        family=true_family,
        tau=true_tau,
        n=sample_size,
        seed=seed,
        student_df=(
            float(true_student_df)
            if true_student_df is not None
            else 4.0
        ),
    )

    train_data, test_data = split_train_test(
        data=simulation.data,
        train_fraction=train_fraction,
        seed=seed,
    )

    fitted_model = fit_pair_copula(
        train_data=train_data,
        controls=controls,
    )

    selected_family = family_name_from_model(
        fitted_model
    )

    selected_rotation = int(
        fitted_model.rotation
    )

    estimated_tau = float(
        fitted_model.tau
    )

    true_parameters = flatten_parameters(
        simulation.model
    )

    estimated_parameters = flatten_parameters(
        fitted_model
    )

    true_df_value = (
        float(true_student_df)
        if true_student_df is not None
        else float("nan")
    )

    estimated_df_value = extract_student_df(
        family=selected_family,
        parameters=estimated_parameters,
    )

    if (
        np.isfinite(true_df_value)
        and np.isfinite(estimated_df_value)
    ):
        student_df_error = (
            estimated_df_value - true_df_value
        )
        student_df_abs_error = abs(student_df_error)
    else:
        student_df_error = float("nan")
        student_df_abs_error = float("nan")

    train_loglik = safe_loglik(
        fitted_model,
        train_data,
    )

    train_bic = safe_bic(
        fitted_model,
        train_data,
    )

    if compute_heldout_loglik:
        test_loglik = safe_loglik(
            fitted_model,
            test_data,
        )
        test_mean_loglik = (
            test_loglik / test_data.shape[0]
        )

        oracle_test_loglik = safe_loglik(
            simulation.model,
            test_data,
        )
        oracle_test_mean_loglik = (
            oracle_test_loglik / test_data.shape[0]
        )

        test_loglik_gap_from_oracle = (
            test_mean_loglik
            - oracle_test_mean_loglik
        )
    else:
        test_loglik = float("nan")
        test_mean_loglik = float("nan")
        oracle_test_loglik = float("nan")
        oracle_test_mean_loglik = float("nan")
        test_loglik_gap_from_oracle = float("nan")

    family_correct = (
        selected_family == true_family
    )

    exact_model_correct = (
        family_correct
        and selected_rotation == int(simulation.rotation)
    )

    run_id = make_run_id(
        scenario_id=scenario_id,
        sample_size=sample_size,
        seed=seed,
    )

    if save_samples:
        run_directory = runs_directory / run_id

        run_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.savez_compressed(
            run_directory / "samples.npz",
            complete=simulation.data,
            train=train_data,
            test=test_data,
        )

        save_model(
            simulation.model,
            run_directory / "true_model.json",
        )

        save_model(
            fitted_model,
            run_directory / "fitted_model.json",
        )

        save_json(
            {
                "run_id": run_id,
                "scenario_id": scenario_id,
                "true_family": true_family,
                "selected_family": selected_family,
                "true_tau": float(true_tau),
                "estimated_tau": estimated_tau,
                "true_student_df": (
                    None
                    if not np.isfinite(true_df_value)
                    else true_df_value
                ),
                "estimated_student_df": (
                    None
                    if not np.isfinite(estimated_df_value)
                    else estimated_df_value
                ),
                "true_rotation": int(simulation.rotation),
                "selected_rotation": selected_rotation,
                "true_parameters": true_parameters,
                "estimated_parameters": estimated_parameters,
                "sample_size": int(sample_size),
                "train_size": int(train_data.shape[0]),
                "test_size": int(test_data.shape[0]),
                "seed": int(seed),
            },
            run_directory / "metadata.json",
        )

    return {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "true_family": true_family,
        "selected_family": selected_family,
        "family_correct": family_correct,
        "true_rotation": int(simulation.rotation),
        "selected_rotation": selected_rotation,
        "exact_model_correct": exact_model_correct,
        "true_tau": float(true_tau),
        "estimated_tau": estimated_tau,
        "tau_error": estimated_tau - float(true_tau),
        "tau_abs_error": abs(
            estimated_tau - float(true_tau)
        ),
        "complete_empirical_tau": empirical_kendall_tau(
            simulation.data
        ),
        "train_empirical_tau": empirical_kendall_tau(
            train_data
        ),
        "test_empirical_tau": empirical_kendall_tau(
            test_data
        ),
        "true_student_df": true_df_value,
        "estimated_student_df": estimated_df_value,
        "student_df_error": student_df_error,
        "student_df_abs_error": student_df_abs_error,
        "true_parameters": json.dumps(true_parameters),
        "estimated_parameters": json.dumps(
            estimated_parameters
        ),
        "number_estimated_parameters": model_parameter_count(
            fitted_model
        ),
        "sample_size": int(sample_size),
        "train_size": int(train_data.shape[0]),
        "test_size": int(test_data.shape[0]),
        "seed": int(seed),
        "train_loglik": train_loglik,
        "train_mean_loglik": (
            train_loglik / train_data.shape[0]
        ),
        "test_loglik": test_loglik,
        "test_mean_loglik": test_mean_loglik,
        "oracle_test_loglik": oracle_test_loglik,
        "oracle_test_mean_loglik": (
            oracle_test_mean_loglik
        ),
        "test_loglik_gap_from_oracle": (
            test_loglik_gap_from_oracle
        ),
        "train_bic": train_bic,
    }


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------


def create_scenario_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize recovery by generating scenario and sample size."""
    return (
        results
        .groupby(
            [
                "scenario_id",
                "true_family",
                "true_tau",
                "true_student_df",
                "sample_size",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            number_runs=("run_id", "count"),
            family_recovery_accuracy=(
                "family_correct",
                "mean",
            ),
            exact_model_recovery_accuracy=(
                "exact_model_correct",
                "mean",
            ),
            mean_tau_abs_error=(
                "tau_abs_error",
                "mean",
            ),
            median_tau_abs_error=(
                "tau_abs_error",
                "median",
            ),
            mean_student_df_abs_error=(
                "student_df_abs_error",
                "mean",
            ),
            median_student_df_abs_error=(
                "student_df_abs_error",
                "median",
            ),
            mean_test_loglik_per_observation=(
                "test_mean_loglik",
                "mean",
            ),
            mean_test_loglik_gap_from_oracle=(
                "test_loglik_gap_from_oracle",
                "mean",
            ),
        )
        .sort_values(
            [
                "true_family",
                "true_tau",
                "true_student_df",
                "sample_size",
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )


def create_family_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize recovery by family and sample size."""
    return (
        results
        .groupby(
            [
                "true_family",
                "sample_size",
            ],
            as_index=False,
        )
        .agg(
            number_runs=("run_id", "count"),
            family_recovery_accuracy=(
                "family_correct",
                "mean",
            ),
            exact_model_recovery_accuracy=(
                "exact_model_correct",
                "mean",
            ),
            mean_tau_abs_error=(
                "tau_abs_error",
                "mean",
            ),
            median_tau_abs_error=(
                "tau_abs_error",
                "median",
            ),
            mean_test_loglik_gap_from_oracle=(
                "test_loglik_gap_from_oracle",
                "mean",
            ),
        )
        .sort_values(
            [
                "true_family",
                "sample_size",
            ]
        )
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    config_path = resolve_project_path(
        args.config
    )

    config = load_yaml(config_path)
    validate_config(config)

    experiment_config = config["experiment"]
    generation_config = config["data_generation"]
    fitting_config = config["fitting"]
    evaluation_config = config["evaluation"]
    output_config = config["output"]

    output_directory = resolve_project_path(
        output_config["directory"]
    )

    runs_directory = output_directory / "runs"

    if output_directory.exists():
        if args.overwrite:
            shutil.rmtree(output_directory)
        else:
            raise FileExistsError(
                f"Output directory already exists:\n"
                f"{output_directory}\n"
                "Run again with --overwrite to replace it."
            )

    output_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    runs_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        config_path,
        output_directory / "experiment_config.yaml",
    )

    scenarios = expand_scenarios(
        generation_config
    )

    sample_sizes = [
        int(value)
        for value in generation_config["sample_sizes"]
    ]

    seeds = [
        int(value)
        for value in generation_config["seeds"]
    ]

    candidate_families = [
        normalize_family_name(str(family))
        for family in fitting_config["candidate_families"]
    ]

    selection_criterion = str(
        fitting_config.get(
            "selection_criterion",
            "bic",
        )
    ).lower()

    allow_rotations = bool(
        fitting_config.get(
            "allow_rotations",
            True,
        )
    )

    train_fraction = float(
        evaluation_config.get(
            "train_fraction",
            0.8,
        )
    )

    compute_heldout_loglik = bool(
        evaluation_config.get(
            "compute_heldout_loglik",
            True,
        )
    )

    save_samples = bool(
        evaluation_config.get(
            "save_generated_samples",
            False,
        )
    )

    controls = build_fit_controls(
        candidate_families=candidate_families,
        selection_criterion=selection_criterion,
        allow_rotations=allow_rotations,
    )

    total_runs = (
        len(scenarios)
        * len(sample_sizes)
        * len(seeds)
    )

    print("=" * 78)
    print(
        experiment_config.get(
            "name",
            "pair_family_recovery",
        )
    )
    print("=" * 78)
    print(f"Configuration:    {config_path}")
    print(f"Output:           {output_directory}")
    print(f"Scenarios:        {len(scenarios)}")
    print(f"Sample sizes:     {sample_sizes}")
    print(f"Seeds:            {seeds}")
    print(f"Candidate set:    {candidate_families}")
    print(f"Criterion:        {selection_criterion}")
    print(f"Allow rotations:  {allow_rotations}")
    print(f"Total runs:       {total_runs}")
    print("=" * 78)

    result_rows: list[dict[str, Any]] = []
    run_number = 0

    results_path = (
        output_directory
        / "pair_family_recovery_results.csv"
    )

    for scenario in scenarios:
        for sample_size in sample_sizes:
            for seed in seeds:
                run_number += 1

                true_family = str(
                    scenario["family"]
                )
                true_tau = float(
                    scenario["tau"]
                )
                true_student_df = scenario[
                    "student_df"
                ]
                scenario_id = str(
                    scenario["scenario_id"]
                )

                df_text = (
                    ""
                    if true_student_df is None
                    else f" df={true_student_df:g}"
                )

                print(
                    f"[{run_number:04d}/{total_runs:04d}] "
                    f"family={true_family:<8} "
                    f"tau={true_tau:.3f}"
                    f"{df_text} "
                    f"n={sample_size} "
                    f"seed={seed}"
                )

                try:
                    result = run_single_experiment(
                        scenario_id=scenario_id,
                        true_family=true_family,
                        true_tau=true_tau,
                        true_student_df=true_student_df,
                        sample_size=sample_size,
                        seed=seed,
                        train_fraction=train_fraction,
                        controls=controls,
                        compute_heldout_loglik=(
                            compute_heldout_loglik
                        ),
                        save_samples=save_samples,
                        runs_directory=runs_directory,
                    )

                    result["status"] = "success"
                    result["error_message"] = ""

                    print(
                        "    selected="
                        f"{result['selected_family']}"
                        f"[rotation="
                        f"{result['selected_rotation']}], "
                        f"tau_hat="
                        f"{result['estimated_tau']:.4f}, "
                        f"family_correct="
                        f"{result['family_correct']}"
                    )

                except Exception as error:
                    result = {
                        "run_id": make_run_id(
                            scenario_id=scenario_id,
                            sample_size=sample_size,
                            seed=seed,
                        ),
                        "scenario_id": scenario_id,
                        "true_family": true_family,
                        "true_tau": true_tau,
                        "true_student_df": (
                            float(true_student_df)
                            if true_student_df is not None
                            else float("nan")
                        ),
                        "sample_size": sample_size,
                        "seed": seed,
                        "status": "failed",
                        "error_message": (
                            f"{type(error).__name__}: {error}"
                        ),
                    }

                    print(
                        "    FAILED: "
                        f"{result['error_message']}"
                    )

                result_rows.append(result)

                pd.DataFrame(
                    result_rows
                ).to_csv(
                    results_path,
                    index=False,
                )

    results = pd.DataFrame(result_rows)

    successful_results = results.loc[
        results["status"] == "success"
    ].copy()

    if successful_results.empty:
        raise RuntimeError(
            "All runs failed. Inspect "
            "pair_family_recovery_results.csv."
        )

    scenario_summary = create_scenario_summary(
        successful_results
    )

    scenario_summary.to_csv(
        output_directory
        / "pair_family_recovery_scenario_summary.csv",
        index=False,
    )

    family_summary = create_family_summary(
        successful_results
    )

    family_summary.to_csv(
        output_directory
        / "pair_family_recovery_family_summary.csv",
        index=False,
    )

    confusion = pd.crosstab(
        successful_results["true_family"],
        successful_results["selected_family"],
        rownames=["true_family"],
        colnames=["selected_family"],
        dropna=False,
    )

    confusion.to_csv(
        output_directory
        / "family_recovery_confusion_matrix.csv"
    )

    failed_runs = int(
        (results["status"] == "failed").sum()
    )

    successful_runs = int(
        (results["status"] == "success").sum()
    )

    save_json(
        {
            "experiment_name": experiment_config.get(
                "name",
                "pair_family_recovery",
            ),
            "number_generating_scenarios": len(
                scenarios
            ),
            "total_requested_runs": int(total_runs),
            "successful_runs": successful_runs,
            "failed_runs": failed_runs,
            "overall_family_recovery_accuracy": float(
                successful_results[
                    "family_correct"
                ].mean()
            ),
            "overall_exact_model_recovery_accuracy": float(
                successful_results[
                    "exact_model_correct"
                ].mean()
            ),
            "overall_mean_tau_abs_error": float(
                successful_results[
                    "tau_abs_error"
                ].mean()
            ),
            "overall_mean_test_loglik_gap_from_oracle": float(
                successful_results[
                    "test_loglik_gap_from_oracle"
                ].mean()
            ),
        },
        output_directory / "experiment_summary.json",
    )

    print()
    print("=" * 78)
    print("Experiment complete")
    print("=" * 78)
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs:     {failed_runs}")
    print()
    print(family_summary.to_string(index=False))
    print()
    print(f"Detailed results: {results_path}")
    print(
        "Scenario summary: "
        f"{output_directory / 'pair_family_recovery_scenario_summary.csv'}"
    )
    print(
        "Family summary:   "
        f"{output_directory / 'pair_family_recovery_family_summary.csv'}"
    )
    print(
        "Confusion matrix: "
        f"{output_directory / 'family_recovery_confusion_matrix.csv'}"
    )


if __name__ == "__main__":
    main()