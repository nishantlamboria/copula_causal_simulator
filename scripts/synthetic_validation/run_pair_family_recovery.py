"""Run the synthetic pair-copula family-recovery experiment.

The experiment:

1. reads a YAML configuration;
2. generates samples from known pair-copula families;
3. splits each sample into training and test observations;
4. selects a fitted pair copula from the configured candidate set;
5. records family recovery, parameter recovery, Kendall-tau error,
   BIC, and held-out log-likelihood;
6. optionally saves the samples and fitted models.

Run from the project root:

    python scripts/synthetic_validation/run_pair_family_recovery.py \
        --config configs/synthetic_validation/pair_family_recovery_smoke.yaml
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
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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
            "Run synthetic pair-copula family and parameter recovery."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the synthetic-validation YAML configuration.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Delete an existing experiment output directory before "
            "running."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_yaml(path: Path) -> dict[str, Any]:
    """Load and validate the top-level YAML object."""
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
    """Validate required configuration sections and values."""
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

    required_generation_fields = {
        "families",
        "kendall_taus",
        "sample_sizes",
        "seeds",
    }

    missing_generation = required_generation_fields.difference(
        generation
    )

    if missing_generation:
        raise ValueError(
            "Missing data_generation fields: "
            + ", ".join(sorted(missing_generation))
        )

    if not generation["families"]:
        raise ValueError(
            "data_generation.families must not be empty."
        )

    if not generation["kendall_taus"]:
        raise ValueError(
            "data_generation.kendall_taus must not be empty."
        )

    if not generation["sample_sizes"]:
        raise ValueError(
            "data_generation.sample_sizes must not be empty."
        )

    if not generation["seeds"]:
        raise ValueError(
            "data_generation.seeds must not be empty."
        )

    for sample_size in generation["sample_sizes"]:
        if not isinstance(sample_size, int) or sample_size < 4:
            raise ValueError(
                "All sample sizes must be integers of at least 4."
            )

    for seed in generation["seeds"]:
        if not isinstance(seed, int) or seed < 0:
            raise ValueError(
                "All seeds must be non-negative integers."
            )

    train_fraction = float(evaluation["train_fraction"])

    if not 0.0 < train_fraction < 1.0:
        raise ValueError(
            "evaluation.train_fraction must lie strictly between "
            "0 and 1."
        )

    candidate_families = fitting.get(
        "candidate_families",
        [],
    )

    if not candidate_families:
        raise ValueError(
            "fitting.candidate_families must not be empty."
        )

    criterion = str(
        fitting.get(
            "selection_criterion",
            "bic",
        )
    ).lower()

    allowed_criteria = {
        "loglik",
        "aic",
        "bic",
        "mbic",
    }

    if criterion not in allowed_criteria:
        raise ValueError(
            "Unsupported selection criterion "
            f"{criterion!r}. Expected one of "
            f"{sorted(allowed_criteria)}."
        )


# ---------------------------------------------------------------------
# pyvinecopulib compatibility helpers
# ---------------------------------------------------------------------


def family_enum(family_name: str):
    """Map a configured family name to the installed family enum."""
    canonical = normalize_family_name(family_name)

    if hasattr(pv, "BicopFamily"):
        try:
            return getattr(
                pv.BicopFamily,
                canonical,
            )
        except AttributeError as error:
            raise ValueError(
                f"Installed pyvinecopulib does not support "
                f"{canonical!r}."
            ) from error

    try:
        return getattr(
            pv,
            canonical,
        )
    except AttributeError as error:
        raise ValueError(
            f"Installed pyvinecopulib does not support "
            f"{canonical!r}."
        ) from error


def family_name_from_model(model: pv.Bicop) -> str:
    """Extract a stable lowercase family name from a fitted model."""
    family = model.family

    enum_name = getattr(
        family,
        "name",
        None,
    )

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
    """Construct fitting controls with version compatibility."""
    family_set = [
        family_enum(name)
        for name in candidate_families
    ]

    arguments = {
        "family_set": family_set,
        "selection_criterion": selection_criterion,
        "parametric_method": "mle",
        "preselect_families": False,
        "num_threads": 1,
    }

    # Recent versions support allow_rotations in the constructor.
    try:
        return pv.FitControlsBicop(
            **arguments,
            allow_rotations=allow_rotations,
        )
    except TypeError:
        controls = pv.FitControlsBicop(**arguments)

        # Older versions may not expose this option. The smoke config
        # requests rotations, which was the historical default.
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
    """Fit and select a pair copula on training pseudo-observations."""
    data = np.asfortranarray(
        train_data,
        dtype=float,
    )

    if hasattr(pv.Bicop, "from_data"):
        return pv.Bicop.from_data(
            data=data,
            controls=controls,
        )

    # Compatibility with older pyvinecopulib releases.
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
    """Create a deterministic shuffled train/test split."""
    n = data.shape[0]

    train_size = int(
        np.floor(
            train_fraction * n
        )
    )

    train_size = max(
        2,
        min(
            train_size,
            n - 2,
        ),
    )

    rng = np.random.default_rng(
        seed + 1_000_003
    )

    indices = rng.permutation(n)

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
    """Calculate empirical Kendall's tau for a two-column sample."""
    result = kendalltau(
        data[:, 0],
        data[:, 1],
    )

    tau = float(getattr(result, "statistic"))

    if not np.isfinite(tau):
        raise RuntimeError(
            "Empirical Kendall's tau is not finite."
        )

    return tau


def flatten_parameters(model: pv.Bicop) -> list[float]:
    """Return model parameters as a JSON-serializable list."""
    parameters = np.asarray(
        model.parameters,
        dtype=float,
    ).reshape(-1)

    return [
        float(value)
        for value in parameters
    ]


def safe_loglik(
    model: pv.Bicop,
    data: np.ndarray,
) -> float:
    """Evaluate the model log-likelihood."""
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
    """Evaluate BIC on the supplied data."""
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


# ---------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------


def save_json(
    payload: dict[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
        )


def save_model(
    model: pv.Bicop,
    path: Path,
) -> None:
    """Save a pyvinecopulib model as JSON."""
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
    family: str,
    tau: float,
    sample_size: int,
    seed: int,
) -> str:
    tau_text = (
        f"{tau:.3f}"
        .replace("-", "minus")
        .replace(".", "p")
    )

    return (
        f"family_{family}"
        f"__tau_{tau_text}"
        f"__n_{sample_size}"
        f"__seed_{seed}"
    )


# ---------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------


def run_single_experiment(
    *,
    true_family: str,
    true_tau: float,
    sample_size: int,
    seed: int,
    train_fraction: float,
    controls: pv.FitControlsBicop,
    compute_heldout_loglik: bool,
    save_samples: bool,
    runs_directory: Path,
) -> dict[str, Any]:
    """Run one ground-truth generation and recovery experiment."""
    simulation = simulate_pair_copula(
        family=true_family,
        tau=true_tau,
        n=sample_size,
        seed=seed,
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

    train_empirical_tau = empirical_kendall_tau(
        train_data
    )

    test_empirical_tau = empirical_kendall_tau(
        test_data
    )

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
    else:
        test_loglik = float("nan")
        test_mean_loglik = float("nan")

    true_parameters = flatten_parameters(
        simulation.model
    )

    estimated_parameters = flatten_parameters(
        fitted_model
    )

    run_id = make_run_id(
        family=true_family,
        tau=true_tau,
        sample_size=sample_size,
        seed=seed,
    )

    run_directory = runs_directory / run_id

    if save_samples:
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
                "true_family": true_family,
                "selected_family": selected_family,
                "true_tau": float(true_tau),
                "estimated_tau": estimated_tau,
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

    result = {
        "run_id": run_id,
        "true_family": true_family,
        "selected_family": selected_family,
        "family_correct": (
            selected_family == true_family
        ),
        "true_rotation": int(simulation.rotation),
        "selected_rotation": selected_rotation,
        "exact_model_correct": (
            selected_family == true_family
            and selected_rotation == int(simulation.rotation)
        ),
        "true_tau": float(true_tau),
        "estimated_tau": estimated_tau,
        "tau_error": (
            estimated_tau - float(true_tau)
        ),
        "tau_abs_error": abs(
            estimated_tau - float(true_tau)
        ),
        "complete_empirical_tau": empirical_kendall_tau(
            simulation.data
        ),
        "train_empirical_tau": train_empirical_tau,
        "test_empirical_tau": test_empirical_tau,
        "true_parameters": json.dumps(
            true_parameters
        ),
        "estimated_parameters": json.dumps(
            estimated_parameters
        ),
        "number_estimated_parameters": int(
            fitted_model.npars
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
        "train_bic": train_bic,
    }

    return result


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------


def create_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Create a compact summary by true family."""
    summary = (
        results
        .groupby(
            "true_family",
            as_index=False,
        )
        .agg(
            number_runs=(
                "run_id",
                "count",
            ),
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
            mean_test_loglik_per_observation=(
                "test_mean_loglik",
                "mean",
            ),
        )
    )

    return summary


# ---------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    config_path = resolve_project_path(
        args.config
    )

    config = load_yaml(
        config_path
    )

    validate_config(
        config
    )

    experiment_config = config["experiment"]
    generation_config = config["data_generation"]
    fitting_config = config["fitting"]
    evaluation_config = config["evaluation"]
    output_config = config["output"]

    output_directory = resolve_project_path(
        output_config["directory"]
    )

    runs_directory = output_directory / "runs"

    if output_directory.exists() and args.overwrite:
        shutil.rmtree(
            output_directory
        )

    if output_directory.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output directory already exists: {output_directory}\n"
            "Use --overwrite to replace it."
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    runs_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    copied_config_path = (
        output_directory / "experiment_config.yaml"
    )

    shutil.copy2(
        config_path,
        copied_config_path,
    )

    true_families = [
        normalize_family_name(family)
        for family in generation_config["families"]
    ]

    true_taus = [
        float(value)
        for value in generation_config["kendall_taus"]
    ]

    sample_sizes = [
        int(value)
        for value in generation_config["sample_sizes"]
    ]

    seeds = [
        int(value)
        for value in generation_config["seeds"]
    ]

    candidate_families = [
        normalize_family_name(family)
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
            True,
        )
    )

    controls = build_fit_controls(
        candidate_families=candidate_families,
        selection_criterion=selection_criterion,
        allow_rotations=allow_rotations,
    )

    total_runs = (
        len(true_families)
        * len(true_taus)
        * len(sample_sizes)
        * len(seeds)
    )

    print("=" * 72)
    print(
        experiment_config.get(
            "name",
            "pair_family_recovery",
        )
    )
    print("=" * 72)
    print(f"Configuration: {config_path}")
    print(f"Output:        {output_directory}")
    print(f"True families: {true_families}")
    print(f"Candidate set: {candidate_families}")
    print(f"Kendall taus:  {true_taus}")
    print(f"Sample sizes:  {sample_sizes}")
    print(f"Seeds:         {seeds}")
    print(f"Criterion:     {selection_criterion}")
    print(f"Rotations:     {allow_rotations}")
    print(f"Total runs:    {total_runs}")
    print("=" * 72)

    result_rows: list[dict[str, Any]] = []

    run_number = 0

    for true_family in true_families:
        for true_tau in true_taus:
            for sample_size in sample_sizes:
                for seed in seeds:
                    run_number += 1

                    print(
                        f"[{run_number:02d}/{total_runs:02d}] "
                        f"family={true_family:<8} "
                        f"tau={true_tau:.3f} "
                        f"n={sample_size} "
                        f"seed={seed}"
                    )

                    try:
                        result = run_single_experiment(
                            true_family=true_family,
                            true_tau=true_tau,
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
                            f"[rotation={result['selected_rotation']}], "
                            f"tau_hat="
                            f"{result['estimated_tau']:.4f}, "
                            f"correct="
                            f"{result['family_correct']}"
                        )

                    except Exception as error:
                        result = {
                            "run_id": make_run_id(
                                family=true_family,
                                tau=true_tau,
                                sample_size=sample_size,
                                seed=seed,
                            ),
                            "true_family": true_family,
                            "true_tau": true_tau,
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

                    result_rows.append(
                        result
                    )

                    # Save after every run so partial progress is retained.
                    pd.DataFrame(
                        result_rows
                    ).to_csv(
                        output_directory
                        / "pair_family_recovery_results.csv",
                        index=False,
                    )

    results = pd.DataFrame(
        result_rows
    )

    successful_results = results.loc[
        results["status"] == "success"
    ].copy()

    if successful_results.empty:
        raise RuntimeError(
            "All synthetic-validation runs failed. "
            "Inspect pair_family_recovery_results.csv."
        )

    summary = create_summary(
        successful_results
    )

    summary.to_csv(
        output_directory
        / "pair_family_recovery_summary.csv",
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

    save_json(
        {
            "experiment_name": experiment_config.get(
                "name",
                "pair_family_recovery",
            ),
            "total_requested_runs": int(total_runs),
            "successful_runs": int(
                (
                    results["status"] == "success"
                ).sum()
            ),
            "failed_runs": int(
                (
                    results["status"] == "failed"
                ).sum()
            ),
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
        },
        output_directory / "experiment_summary.json",
    )

    print("\n" + "=" * 72)
    print("Experiment complete")
    print("=" * 72)
    print(summary.to_string(index=False))
    print()
    print(
        "Results:   "
        f"{output_directory / 'pair_family_recovery_results.csv'}"
    )
    print(
        "Summary:   "
        f"{output_directory / 'pair_family_recovery_summary.csv'}"
    )
    print(
        "Confusion: "
        f"{output_directory / 'family_recovery_confusion_matrix.csv'}"
    )


if __name__ == "__main__":
    main()