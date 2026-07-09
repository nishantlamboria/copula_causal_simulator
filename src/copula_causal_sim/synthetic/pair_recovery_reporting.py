"""Reporting utilities for synthetic pair-copula recovery experiments.

The functions in this module are deliberately separated from the command-line
script so that result validation, model reconstruction, scenario selection,
and summary calculations can be tested independently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyvinecopulib as pv

from copula_causal_sim.synthetic.pair_copula import normalize_family_name


FAMILY_ORDER = (
    "indep",
    "gaussian",
    "student",
    "clayton",
    "gumbel",
    "frank",
)

FAMILY_LABELS = {
    "indep": "Independence",
    "gaussian": "Gaussian",
    "student": "Student t",
    "clayton": "Clayton",
    "gumbel": "Gumbel",
    "frank": "Frank",
}

REQUIRED_RESULT_COLUMNS = {
    "run_id",
    "scenario_id",
    "true_family",
    "selected_family",
    "family_correct",
    "true_rotation",
    "selected_rotation",
    "true_tau",
    "estimated_tau",
    "tau_abs_error",
    "true_student_df",
    "estimated_student_df",
    "student_df_abs_error",
    "true_parameters",
    "estimated_parameters",
    "sample_size",
    "seed",
    "test_loglik_gap_from_oracle",
    "status",
}


@dataclass(frozen=True)
class RepresentativeCase:
    """Specification for one true-versus-fitted copula figure."""

    slug: str
    family: str
    tau: float
    sample_size: int
    student_df: float | None = None
    prefer_correct: bool | None = True
    preferred_selected_family: str | None = None


DEFAULT_REPRESENTATIVE_CASES = (
    RepresentativeCase(
        slug="gaussian_tau_0p5_n1000",
        family="gaussian",
        tau=0.5,
        sample_size=1000,
    ),
    RepresentativeCase(
        slug="clayton_tau_0p5_n1000",
        family="clayton",
        tau=0.5,
        sample_size=1000,
    ),
    RepresentativeCase(
        slug="gumbel_tau_0p5_n1000",
        family="gumbel",
        tau=0.5,
        sample_size=1000,
    ),
    RepresentativeCase(
        slug="frank_tau_0p2_n250_challenging",
        family="frank",
        tau=0.2,
        sample_size=250,
        prefer_correct=False,
    ),
    RepresentativeCase(
        slug="student_df4_tau_0p5_n1000",
        family="student",
        tau=0.5,
        sample_size=1000,
        student_df=4.0,
    ),
    RepresentativeCase(
        slug="student_df10_tau_0p5_n1000_challenging",
        family="student",
        tau=0.5,
        sample_size=1000,
        student_df=10.0,
        prefer_correct=False,
        preferred_selected_family="gaussian",
    ),
)


def read_results(path: str | Path) -> pd.DataFrame:
    """Read, validate, and normalize a pair-recovery result table."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Results file does not exist: {path}")

    results = pd.read_csv(path)
    validate_results(results)
    results = results.loc[results["status"] == "success"].copy()

    if results.empty:
        raise ValueError("The result table contains no successful runs.")

    results["family_correct"] = coerce_boolean_series(
        results["family_correct"],
        name="family_correct",
    )

    for column in ("true_family", "selected_family"):
        results[column] = results[column].map(normalize_family_name)

    numeric_columns = (
        "true_tau",
        "estimated_tau",
        "tau_abs_error",
        "true_student_df",
        "estimated_student_df",
        "student_df_abs_error",
        "sample_size",
        "seed",
        "true_rotation",
        "selected_rotation",
        "test_loglik_gap_from_oracle",
    )

    for column in numeric_columns:
        results[column] = pd.to_numeric(
            results[column],
            errors="coerce",
        )

    return results.reset_index(drop=True)


def validate_results(results: pd.DataFrame) -> None:
    """Validate the columns required by the reporting pipeline."""
    missing = REQUIRED_RESULT_COLUMNS.difference(results.columns)

    if missing:
        raise ValueError(
            "The result table is missing required columns: "
            + ", ".join(sorted(missing))
        )


def coerce_boolean_series(
    values: pd.Series,
    *,
    name: str,
) -> pd.Series:
    """Convert common CSV boolean representations to ``bool``."""
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }

    converted = (
        values.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )

    if converted.isna().any():
        invalid = sorted(
            values.loc[converted.isna()].astype(str).unique().tolist()
        )
        raise ValueError(
            f"Could not interpret values in {name!r} as booleans: "
            f"{invalid}"
        )

    return converted.astype(bool)


def ordered_families(values: Iterable[str]) -> list[str]:
    """Return present family names in the canonical display order."""
    present = {normalize_family_name(value) for value in values}
    ordered = [family for family in FAMILY_ORDER if family in present]
    extras = sorted(present.difference(FAMILY_ORDER))
    return ordered + extras


def family_label(family: str) -> str:
    """Return the presentation label for a canonical family name."""
    canonical = normalize_family_name(family)
    return FAMILY_LABELS.get(canonical, canonical.title())


def parse_parameter_vector(value: object) -> np.ndarray:
    """Parse a JSON-encoded parameter vector from a result row."""
    if isinstance(value, np.ndarray):
        parsed = value.astype(float, copy=False).reshape(-1)
    elif isinstance(value, (list, tuple)):
        parsed = np.asarray(value, dtype=float).reshape(-1)
    elif pd.isna(value):
        parsed = np.empty(0, dtype=float)
    else:
        try:
            parsed_value = json.loads(str(value))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Could not parse copula parameters from {value!r}."
            ) from error
        parsed = np.asarray(parsed_value, dtype=float).reshape(-1)

    if not np.all(np.isfinite(parsed)):
        raise ValueError("Copula parameters contain non-finite values.")

    return parsed


def family_enum(family: str):
    """Return the installed pyvinecopulib family enum."""
    canonical = normalize_family_name(family)

    if hasattr(pv, "BicopFamily"):
        return getattr(pv.BicopFamily, canonical)

    return getattr(pv, canonical)


def parameter_matrix(
    family: str,
    parameters: np.ndarray,
) -> np.ndarray:
    """Convert a flat result vector into pyvinecopulib's matrix shape."""
    canonical = normalize_family_name(family)
    flat = np.asarray(parameters, dtype=float).reshape(-1)

    if canonical == "indep":
        if flat.size != 0:
            raise ValueError(
                "The independence copula must have no parameters."
            )
        return np.empty((0, 0), dtype=float, order="F")

    expected_size = 2 if canonical == "student" else 1

    if flat.size != expected_size:
        raise ValueError(
            f"Family {canonical!r} requires {expected_size} parameters, "
            f"but {flat.size} were supplied."
        )

    return np.asfortranarray(flat.reshape(expected_size, 1))


def reconstruct_model(
    family: str,
    rotation: int,
    parameters: object,
) -> pv.Bicop:
    """Reconstruct a fitted or true ``Bicop`` from a result row."""
    canonical = normalize_family_name(family)
    vector = parse_parameter_vector(parameters)
    matrix = parameter_matrix(canonical, vector)

    if hasattr(pv.Bicop, "from_family"):
        return pv.Bicop.from_family(
            family=family_enum(canonical),
            rotation=int(rotation),
            parameters=matrix,
        )

    return pv.Bicop(
        family_enum(canonical),
        int(rotation),
        matrix,
    )


def confusion_tables(
    results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return count and row-normalized family confusion matrices."""
    families = ordered_families(
        pd.concat(
            [results["true_family"], results["selected_family"]],
            ignore_index=True,
        )
    )

    counts = pd.crosstab(
        results["true_family"],
        results["selected_family"],
        dropna=False,
    ).reindex(index=families, columns=families, fill_value=0)

    row_totals = counts.sum(axis=1).replace(0, np.nan)
    proportions = counts.div(row_totals, axis=0).fillna(0.0)

    counts.index.name = "true_family"
    counts.columns.name = "selected_family"
    proportions.index.name = "true_family"
    proportions.columns.name = "selected_family"

    return counts, proportions


def family_sample_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize recovery and fit quality by family and sample size."""
    summary = (
        results.groupby(
            ["true_family", "sample_size"],
            as_index=False,
        )
        .agg(
            number_runs=("run_id", "size"),
            family_recovery_accuracy=("family_correct", "mean"),
            mean_tau_abs_error=("tau_abs_error", "mean"),
            median_tau_abs_error=("tau_abs_error", "median"),
            mean_test_loglik_gap_from_oracle=(
                "test_loglik_gap_from_oracle",
                "mean",
            ),
        )
    )

    family_rank = {
        family: rank for rank, family in enumerate(FAMILY_ORDER)
    }
    summary["_family_rank"] = summary["true_family"].map(
        family_rank
    ).fillna(len(FAMILY_ORDER))

    return (
        summary.sort_values(["_family_rank", "sample_size"])
        .drop(columns="_family_rank")
        .reset_index(drop=True)
    )


def scenario_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize recovery for each complete generating scenario."""
    summary = (
        results.groupby(
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
            number_runs=("run_id", "size"),
            family_recovery_accuracy=("family_correct", "mean"),
            mean_tau_abs_error=("tau_abs_error", "mean"),
            median_tau_abs_error=("tau_abs_error", "median"),
            mean_estimated_tau=("estimated_tau", "mean"),
            mean_student_df_abs_error=(
                "student_df_abs_error",
                "mean",
            ),
            mean_test_loglik_gap_from_oracle=(
                "test_loglik_gap_from_oracle",
                "mean",
            ),
        )
    )

    family_rank = {
        family: rank for rank, family in enumerate(FAMILY_ORDER)
    }
    summary["_family_rank"] = summary["true_family"].map(
        family_rank
    ).fillna(len(FAMILY_ORDER))

    return (
        summary.sort_values(
            [
                "_family_rank",
                "true_student_df",
                "true_tau",
                "sample_size",
            ],
            na_position="last",
        )
        .drop(columns="_family_rank")
        .reset_index(drop=True)
    )


def select_representative_run(
    results: pd.DataFrame,
    case: RepresentativeCase,
) -> pd.Series:
    """Select one deterministic representative run for a plot.

    Selection first applies the requested scenario restrictions. When a
    preferred correctness state or selected family is available, it is used.
    The remaining candidates are ranked by distance to the median absolute
    Kendall-tau error and then by seed.
    """
    family = normalize_family_name(case.family)

    mask = (
        (results["true_family"] == family)
        & np.isclose(results["true_tau"], case.tau)
        & (results["sample_size"] == case.sample_size)
    )

    if family == "student":
        if case.student_df is None:
            raise ValueError(
                "A Student representative case requires student_df."
            )
        mask &= np.isclose(
            results["true_student_df"],
            case.student_df,
            equal_nan=False,
        )

    candidates = results.loc[mask].copy()

    if candidates.empty:
        raise ValueError(
            "No result rows match representative case "
            f"{case.slug!r}."
        )

    if case.preferred_selected_family is not None:
        preferred_family = normalize_family_name(
            case.preferred_selected_family
        )
        preferred = candidates.loc[
            candidates["selected_family"] == preferred_family
        ]
        if not preferred.empty:
            candidates = preferred

    if case.prefer_correct is not None:
        preferred = candidates.loc[
            candidates["family_correct"] == case.prefer_correct
        ]
        if not preferred.empty:
            candidates = preferred

    median_error = float(candidates["tau_abs_error"].median())
    candidates["_median_distance"] = (
        candidates["tau_abs_error"] - median_error
    ).abs()

    selected = candidates.sort_values(
        ["_median_distance", "tau_abs_error", "seed", "run_id"]
    ).iloc[0]

    return selected.drop(labels="_median_distance")


def density_grid(
    model: pv.Bicop,
    *,
    grid_size: int = 140,
    epsilon: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a copula density on a regular open-unit-square grid."""
    if grid_size < 20:
        raise ValueError("grid_size must be at least 20.")

    if not 0.0 < epsilon < 0.25:
        raise ValueError("epsilon must lie between 0 and 0.25.")

    coordinates = np.linspace(
        epsilon,
        1.0 - epsilon,
        grid_size,
    )
    u1, u2 = np.meshgrid(coordinates, coordinates, indexing="xy")
    points = np.asfortranarray(
        np.column_stack([u1.ravel(), u2.ravel()]),
        dtype=float,
    )
    density = np.asarray(model.pdf(points), dtype=float).reshape(
        grid_size,
        grid_size,
    )

    if not np.all(np.isfinite(density)):
        raise RuntimeError("Copula density evaluation returned non-finite values.")

    return u1, u2, density


def summary_statistics(results: pd.DataFrame) -> dict[str, float | int]:
    """Calculate the headline values used in the Markdown report."""
    total_runs = int(len(results))
    counts, proportions = confusion_tables(results)
    _ = counts

    student = results.loc[results["true_family"] == "student"]
    student_df4 = student.loc[np.isclose(student["true_student_df"], 4.0)]
    student_df10 = student.loc[
        np.isclose(student["true_student_df"], 10.0)
    ]

    n1000 = results.loc[results["sample_size"] == 1000]

    return {
        "total_runs": total_runs,
        "overall_family_accuracy": float(results["family_correct"].mean()),
        "overall_tau_mae": float(results["tau_abs_error"].mean()),
        "accuracy_n1000": float(n1000["family_correct"].mean()),
        "tau_mae_n1000": float(n1000["tau_abs_error"].mean()),
        "student_accuracy": float(student["family_correct"].mean()),
        "student_df4_accuracy": float(
            student_df4["family_correct"].mean()
        ),
        "student_df10_accuracy": float(
            student_df10["family_correct"].mean()
        ),
        "student_to_gaussian_rate": float(
            (student["selected_family"] == "gaussian").mean()
        ),
        "minimum_family_recall": float(
            proportions.to_numpy().diagonal().min()
        ),
    }
