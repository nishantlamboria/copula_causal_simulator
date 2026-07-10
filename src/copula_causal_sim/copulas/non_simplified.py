"""Adaptive quantile-binned conditional copulas for two-parent D-vines.

The module relaxes the simplifying assumption for an explicit D-vine order
``A -- B -- X``.  The first-tree copulas remain global, while the second-tree
copula of ``F(A|B)`` and ``F(X|B)`` may vary across equal-frequency bins of
``U_B``.

The candidate number of bins is selected by held-out conditional
log-likelihood.  A binned model is accepted only when it improves on the
one-bin simplified model by a user-specified minimum margin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

try:
    import pyvinecopulib as pv
except ImportError as exc:  # pragma: no cover - exercised by package import
    raise ImportError(
        "pyvinecopulib is required for adaptive conditional copulas."
    ) from exc


FloatArray = np.ndarray


def clip_u(values: Any, eps: float = 1e-6) -> FloatArray:
    """Return finite pseudo-observations inside the open unit interval."""
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("Pseudo-observations contain non-finite values.")
    return np.clip(array, eps, 1.0 - eps)


def model_family_name(model: Any) -> str:
    """Return a stable lowercase family name."""
    family = model.family
    name = getattr(family, "name", None)
    if isinstance(name, str):
        return name.lower()
    return str(family).split(".")[-1].lower()


def model_npars(model: Any) -> float:
    """Return the effective number of model parameters."""
    value = getattr(model, "npars", 0.0)
    if callable(value):
        value = value()
    return float(value)


def model_summary(model: Any) -> dict[str, Any]:
    """Return JSON/pickle-friendly metadata for one pair copula."""
    return {
        "family": model_family_name(model),
        "rotation": int(getattr(model, "rotation", 0)),
        "parameters": np.asarray(model.parameters, dtype=float).tolist(),
        "tau": float(model.tau),
        "npars": model_npars(model),
    }


def pointwise_log_pdf(
    model: Any,
    data: FloatArray,
    *,
    eps: float = 1e-6,
    density_floor: float = 1e-300,
) -> FloatArray:
    """Evaluate protected pointwise log copula densities."""
    array = np.asfortranarray(clip_u(data, eps=eps), dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"Expected an n x 2 array, received {array.shape}.")
    density = np.asarray(model.pdf(array), dtype=float).reshape(-1)
    return np.log(np.clip(density, density_floor, None))


def compute_conditional_coordinates(
    *,
    c_first_second: Any,
    c_second_child: Any,
    data: pd.DataFrame,
    first_parent: str,
    second_parent: str,
    child: str,
    clip_eps: float = 1e-6,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Compute ``U_B``, ``F(A|B)``, and ``F(X|B)`` for order ``A--B--X``."""
    required = {first_parent, second_parent, child}
    missing = required.difference(data.columns)
    if missing:
        raise KeyError(
            "Missing variables required for conditional coordinates: "
            + ", ".join(sorted(missing))
        )

    u_first = clip_u(data[first_parent].to_numpy(dtype=float), eps=clip_eps)
    u_second = clip_u(data[second_parent].to_numpy(dtype=float), eps=clip_eps)
    u_child = clip_u(data[child].to_numpy(dtype=float), eps=clip_eps)

    first_second = np.asfortranarray(
        np.column_stack([u_first, u_second]), dtype=float
    )
    second_child = np.asfortranarray(
        np.column_stack([u_second, u_child]), dtype=float
    )

    w_first = clip_u(c_first_second.hfunc2(first_second), eps=clip_eps)
    w_child = clip_u(c_second_child.hfunc1(second_child), eps=clip_eps)
    return u_second, w_first, w_child


def normalize_candidate_bin_counts(values: Iterable[int]) -> list[int]:
    """Validate, sort, and deduplicate candidate bin counts."""
    result: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError("Candidate bin counts must be integers.")
        integer = int(value)
        if integer < 1:
            raise ValueError("Candidate bin counts must be positive.")
        if integer not in result:
            result.append(integer)
    if not result:
        raise ValueError("At least one candidate bin count is required.")
    if 1 not in result:
        result.append(1)
    return sorted(result)


def make_quantile_bin_edges(
    conditioning: FloatArray,
    *,
    n_bins: int,
    minimum_bin_size: int,
    clip_eps: float = 1e-6,
) -> tuple[FloatArray, FloatArray, list[int]]:
    """Create training-only equal-frequency bin boundaries and assignments.

    The outer boundaries are fixed at zero and one so that future observations
    outside the empirical training range are assigned to an edge bin.
    """
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)):
        raise TypeError("n_bins must be an integer.")
    if int(n_bins) < 1:
        raise ValueError("n_bins must be positive.")
    if isinstance(minimum_bin_size, bool) or not isinstance(
        minimum_bin_size, (int, np.integer)
    ):
        raise TypeError("minimum_bin_size must be an integer.")
    if int(minimum_bin_size) < 1:
        raise ValueError("minimum_bin_size must be positive.")

    values = clip_u(conditioning, eps=clip_eps).reshape(-1)
    if values.size < int(n_bins) * int(minimum_bin_size):
        raise ValueError(
            f"{values.size} observations cannot support {n_bins} bins with "
            f"minimum_bin_size={minimum_bin_size}."
        )

    quantiles = np.linspace(0.0, 1.0, int(n_bins) + 1)
    edges = np.asarray(np.quantile(values, quantiles), dtype=float)
    edges[0] = 0.0
    edges[-1] = 1.0

    if len(edges) != int(n_bins) + 1 or np.any(np.diff(edges) <= 0.0):
        raise ValueError(
            "Quantile boundaries are not distinct; the requested binned "
            "conditional model is not estimable."
        )

    assignments = assign_quantile_bins(values, edges)
    counts = np.bincount(assignments, minlength=int(n_bins)).astype(int).tolist()
    if any(count < int(minimum_bin_size) for count in counts):
        raise ValueError(
            f"Bin counts {counts} violate minimum_bin_size={minimum_bin_size}."
        )
    return edges, assignments, counts


def assign_quantile_bins(values: FloatArray, edges: FloatArray) -> FloatArray:
    """Assign values to half-open bins, including one in the final bin."""
    boundaries = np.asarray(edges, dtype=float).reshape(-1)
    if boundaries.size < 2 or np.any(np.diff(boundaries) <= 0.0):
        raise ValueError("Bin edges must be a strictly increasing vector.")
    array = np.asarray(values, dtype=float).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError("Conditioning values contain non-finite entries.")
    return np.clip(
        np.searchsorted(boundaries[1:-1], array, side="right"),
        0,
        boundaries.size - 2,
    ).astype(int)


def fit_fixed_family_rotation_bicop(
    data: FloatArray,
    *,
    family: Any,
    rotation: int,
    clip_eps: float = 1e-6,
) -> Any:
    """Fit parameters while keeping family and rotation fixed."""
    array = np.asfortranarray(clip_u(data, eps=clip_eps), dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"Expected an n x 2 array, received {array.shape}.")

    model = pv.Bicop.from_family(
        family=family,
        rotation=int(rotation),
        var_types=["c", "c"],
    )
    controls = pv.FitControlsBicop(
        parametric_method="mle",
        selection_criterion="loglik",
        preselect_families=False,
        num_threads=1,
    )
    model.fit(array, controls=controls)
    return model


@dataclass
class FittedBinnedConditionalCopula:
    """One fixed family/rotation fitted separately across quantile bins."""

    family: str
    rotation: int
    bin_edges: FloatArray
    bin_counts: list[int]
    models: list[Any]

    @property
    def n_bins(self) -> int:
        return len(self.models)

    def bin_indices(self, conditioning: FloatArray) -> FloatArray:
        return assign_quantile_bins(conditioning, self.bin_edges)

    def loglik_values(
        self,
        *,
        w_first: FloatArray,
        w_child: FloatArray,
        conditioning: FloatArray,
        clip_eps: float = 1e-6,
    ) -> FloatArray:
        indices = self.bin_indices(conditioning)
        output = np.empty(len(indices), dtype=float)
        first = clip_u(w_first, eps=clip_eps)
        child = clip_u(w_child, eps=clip_eps)
        for index, model in enumerate(self.models):
            mask = indices == index
            if np.any(mask):
                output[mask] = pointwise_log_pdf(
                    model,
                    np.column_stack([first[mask], child[mask]]),
                    eps=clip_eps,
                )
        return output

    def tau_at(self, conditioning: FloatArray) -> FloatArray:
        indices = self.bin_indices(conditioning)
        taus = np.asarray([float(model.tau) for model in self.models], dtype=float)
        return taus[indices]

    def summaries(self) -> list[dict[str, Any]]:
        return [model_summary(model) for model in self.models]


def fit_binned_conditional_copula(
    *,
    w_first: FloatArray,
    w_child: FloatArray,
    conditioning: FloatArray,
    family: Any,
    rotation: int,
    n_bins: int,
    minimum_bin_size: int,
    clip_eps: float = 1e-6,
) -> FittedBinnedConditionalCopula:
    """Fit a piecewise-constant conditional copula."""
    first = clip_u(w_first, eps=clip_eps).reshape(-1)
    child = clip_u(w_child, eps=clip_eps).reshape(-1)
    conditioning_values = clip_u(conditioning, eps=clip_eps).reshape(-1)
    if not (len(first) == len(child) == len(conditioning_values)):
        raise ValueError("Conditional-coordinate arrays must have equal length.")

    edges, assignments, counts = make_quantile_bin_edges(
        conditioning_values,
        n_bins=int(n_bins),
        minimum_bin_size=int(minimum_bin_size),
        clip_eps=clip_eps,
    )
    models: list[Any] = []
    for bin_index in range(int(n_bins)):
        mask = assignments == bin_index
        model = fit_fixed_family_rotation_bicop(
            np.column_stack([first[mask], child[mask]]),
            family=family,
            rotation=int(rotation),
            clip_eps=clip_eps,
        )
        models.append(model)

    family_name = model_family_name(models[0]) if models else str(family)
    return FittedBinnedConditionalCopula(
        family=family_name,
        rotation=int(rotation),
        bin_edges=edges,
        bin_counts=counts,
        models=models,
    )


@dataclass
class ConditionalBinSelectionResult:
    """Diagnostics and decision from held-out bin-count selection."""

    selected_number_bins: int
    family: str
    family_enum: Any
    rotation: int
    candidate_scores: dict[str, float | None]
    candidate_statuses: dict[str, str]
    candidate_errors: dict[str, str | None]
    simplified_score: float | None
    selected_score: float | None
    raw_best_number_bins: int
    raw_best_score: float | None
    score_improvement_over_simplified: float | None
    minimum_score_improvement: float
    method: str
    training_size: int | None
    validation_size: int | None


def select_number_of_quantile_bins(
    *,
    fitted_training_order: Any,
    training_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    candidate_bin_counts: Sequence[int],
    minimum_bin_size: int,
    minimum_score_improvement: float,
    clip_eps: float = 1e-6,
) -> ConditionalBinSelectionResult:
    """Select the bin count using mean held-out conditional log-likelihood."""
    candidates = normalize_candidate_bin_counts(candidate_bin_counts)
    if validation_df.empty:
        raise ValueError("Validation data must not be empty.")
    if float(minimum_score_improvement) < 0.0:
        raise ValueError("minimum_score_improvement must be non-negative.")

    global_model = fitted_training_order.c_first_child_given_second
    family_enum = global_model.family
    family = model_family_name(global_model)
    rotation = int(global_model.rotation)

    train_conditioning, train_first, train_child = compute_conditional_coordinates(
        c_first_second=fitted_training_order.c_first_second,
        c_second_child=fitted_training_order.c_second_child,
        data=training_df,
        first_parent=fitted_training_order.first_parent,
        second_parent=fitted_training_order.second_parent,
        child=fitted_training_order.child,
        clip_eps=clip_eps,
    )
    validation_conditioning, validation_first, validation_child = (
        compute_conditional_coordinates(
            c_first_second=fitted_training_order.c_first_second,
            c_second_child=fitted_training_order.c_second_child,
            data=validation_df,
            first_parent=fitted_training_order.first_parent,
            second_parent=fitted_training_order.second_parent,
            child=fitted_training_order.child,
            clip_eps=clip_eps,
        )
    )

    candidate_scores: dict[str, float | None] = {
        str(value): None for value in candidates
    }
    candidate_statuses: dict[str, str] = {
        str(value): "not_evaluated" for value in candidates
    }
    candidate_errors: dict[str, str | None] = {
        str(value): None for value in candidates
    }

    simplified_values = pointwise_log_pdf(
        global_model,
        np.column_stack([validation_first, validation_child]),
        eps=clip_eps,
    )
    simplified_score = float(np.mean(simplified_values))
    candidate_scores["1"] = simplified_score
    candidate_statuses["1"] = "ok"
    successful: list[tuple[int, float]] = [(1, simplified_score)]

    for n_bins in candidates:
        if n_bins == 1:
            continue
        key = str(n_bins)
        try:
            fitted = fit_binned_conditional_copula(
                w_first=train_first,
                w_child=train_child,
                conditioning=train_conditioning,
                family=family_enum,
                rotation=rotation,
                n_bins=n_bins,
                minimum_bin_size=minimum_bin_size,
                clip_eps=clip_eps,
            )
            score = float(
                np.mean(
                    fitted.loglik_values(
                        w_first=validation_first,
                        w_child=validation_child,
                        conditioning=validation_conditioning,
                        clip_eps=clip_eps,
                    )
                )
            )
            if not np.isfinite(score):
                raise RuntimeError("Candidate validation score is not finite.")
            candidate_scores[key] = score
            candidate_statuses[key] = "ok"
            successful.append((n_bins, score))
        except Exception as exc:
            candidate_statuses[key] = "failed"
            candidate_errors[key] = f"{type(exc).__name__}: {exc}"

    successful.sort(key=lambda item: (-item[1], item[0]))
    raw_best_bins, raw_best_score = successful[0]
    improvement = float(raw_best_score - simplified_score)
    if raw_best_bins > 1 and improvement >= float(minimum_score_improvement):
        selected_bins = int(raw_best_bins)
        selected_score = float(raw_best_score)
    else:
        selected_bins = 1
        selected_score = simplified_score

    return ConditionalBinSelectionResult(
        selected_number_bins=selected_bins,
        family=family,
        family_enum=family_enum,
        rotation=rotation,
        candidate_scores=candidate_scores,
        candidate_statuses=candidate_statuses,
        candidate_errors=candidate_errors,
        simplified_score=simplified_score,
        selected_score=selected_score,
        raw_best_number_bins=int(raw_best_bins),
        raw_best_score=float(raw_best_score),
        score_improvement_over_simplified=improvement,
        minimum_score_improvement=float(minimum_score_improvement),
        method="heldout_conditional_loglik_with_minimum_improvement",
        training_size=len(training_df),
        validation_size=len(validation_df),
    )


def selected_bin_diagnostics(
    *,
    model: FittedBinnedConditionalCopula,
    w_first: FloatArray,
    w_child: FloatArray,
    conditioning: FloatArray,
) -> list[dict[str, Any]]:
    """Summarize empirical and fitted conditional dependence by bin."""
    indices = model.bin_indices(conditioning)
    global_tau = float(kendalltau(w_first, w_child).statistic)
    rows: list[dict[str, Any]] = []
    for index, fitted_model in enumerate(model.models):
        mask = indices == index
        empirical_tau = (
            float(kendalltau(w_first[mask], w_child[mask]).statistic)
            if np.sum(mask) >= 2
            else float("nan")
        )
        rows.append(
            {
                "bin_index": int(index),
                "bin_lower": float(model.bin_edges[index]),
                "bin_upper": float(model.bin_edges[index + 1]),
                "bin_center": float(
                    0.5 * (model.bin_edges[index] + model.bin_edges[index + 1])
                ),
                "bin_count": int(np.sum(mask)),
                "empirical_conditional_tau": empirical_tau,
                "fitted_conditional_tau": float(fitted_model.tau),
                "global_empirical_conditional_tau": global_tau,
            }
        )
    return rows
