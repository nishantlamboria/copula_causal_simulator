from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
import pickle
import re
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

try:
    import pyvinecopulib as pv
except ImportError as exc:
    raise ImportError(
        "pyvinecopulib is not installed. Install it with:\n\n"
        "    pip install pyvinecopulib\n"
    ) from exc

from copula_causal_sim.copulas.non_simplified import (
    FittedBinnedConditionalCopula,
    compute_conditional_coordinates,
    fit_binned_conditional_copula,
    fit_fixed_family_rotation_bicop,
    normalize_candidate_bin_counts,
    select_number_of_quantile_bins,
    selected_bin_diagnostics,
)


OrderStrategy = Literal[
    "heldout_conditional_loglik",
    "bic",
    "fixed",
]
ConditionalModelStrategy = Literal[
    "simplified",
    "adaptive_quantile_bins",
]


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _safe_filename(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_")


def _enum_to_string(value: Any) -> str:
    if hasattr(value, "name"):
        return str(value.name)
    return str(value).split(".")[-1]


def _parameters_to_list(parameters: Any) -> list:
    try:
        return np.asarray(parameters, dtype=float).tolist()
    except Exception:
        return []


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    except Exception:
        return None


def _safe_metric(model: Any, metric_name: str, data: np.ndarray) -> float | None:
    try:
        method = getattr(model, metric_name)
        return _safe_float(method(data))
    except Exception:
        try:
            method = getattr(model, metric_name)
            return _safe_float(method())
        except Exception:
            return None


def _model_npars(model: Any) -> float:
    value = getattr(model, "npars", None)
    if callable(value):
        value = value()
    value = _safe_float(value)
    return 0.0 if value is None else value


def _model_loglik(model: Any, data: np.ndarray) -> float:
    value = _safe_metric(model, "loglik", data)
    return 0.0 if value is None else value


def _clip_u(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), eps, 1.0 - eps)


def _order_key(first_parent: str, second_parent: str, child: str) -> str:
    return f"{first_parent}--{second_parent}--{child}"


def _canonical_parent_pair(parent_1: str, parent_2: str) -> tuple[str, str]:
    if parent_1 == parent_2:
        raise ValueError("The two parent variables must be distinct.")
    return tuple(sorted((str(parent_1), str(parent_2))))


def _validate_order_strategy(order_strategy: str) -> OrderStrategy:
    normalized = str(order_strategy).strip().lower()
    allowed = {
        "heldout_conditional_loglik",
        "bic",
        "fixed",
    }
    if normalized not in allowed:
        raise ValueError(
            f"Unsupported order strategy {order_strategy!r}. "
            f"Expected one of {sorted(allowed)}."
        )
    return normalized  # type: ignore[return-value]


def _validate_conditional_model_strategy(
    strategy: str,
) -> ConditionalModelStrategy:
    normalized = str(strategy).strip().lower()
    allowed = {"simplified", "adaptive_quantile_bins"}
    if normalized not in allowed:
        raise ValueError(
            f"Unsupported conditional model strategy {strategy!r}. "
            f"Expected one of {sorted(allowed)}."
        )
    return normalized  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Pair-copula fitting helpers
# ---------------------------------------------------------------------------


def _resolve_bicop_family(name: str):
    """Resolve a family enum across pyvinecopulib naming conventions."""
    family_name = str(name).strip().lower()

    for candidate in (family_name, family_name.upper()):
        family = getattr(pv.BicopFamily, candidate, None)
        if family is not None:
            return family

    raise RuntimeError(
        f"The installed pyvinecopulib version does not expose family "
        f"{family_name!r}."
    )


def build_bicop_fit_controls(
    selection_criterion: str = "bic",
    allow_rotations: bool = True,
    num_threads: int = 1,
):
    family_names = [
        "indep",
        "gaussian",
        "student",
        "clayton",
        "gumbel",
        "frank",
    ]

    family_set = [_resolve_bicop_family(name) for name in family_names]

    controls = pv.FitControlsBicop(
        family_set=family_set,
        selection_criterion=selection_criterion,
        allow_rotations=allow_rotations,
        num_threads=num_threads,
    )

    return controls, family_names


def _fit_bicop(data: np.ndarray, controls: Any):
    data = np.asfortranarray(data, dtype=float)
    return pv.Bicop.from_data(
        data,
        controls=controls,
        var_types=["c", "c"],
    )


def _save_bicop(model: Any, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.to_file(str(path))
    return str(path)


def _bicop_summary(model: Any) -> dict[str, Any]:
    return {
        "family": _enum_to_string(model.family),
        "rotation": int(model.rotation) if hasattr(model, "rotation") else None,
        "parameters": _parameters_to_list(model.parameters),
        "npars": _safe_float(getattr(model, "npars", None)),
        "tau": _safe_float(getattr(model, "tau", None)),
    }


# ---------------------------------------------------------------------------
# Explicit three-variable D-vine representation
# ---------------------------------------------------------------------------


@dataclass
class FittedExplicitDvineOrder:
    child: str
    first_parent: str
    second_parent: str
    c_first_second: Any
    c_second_child: Any
    c_first_child_given_second: Any
    data_first_second: np.ndarray
    data_second_child: np.ndarray
    data_first_child_given_second: np.ndarray
    n_obs: int
    npars: float
    loglik: float
    aic: float
    bic: float

    @property
    def order(self) -> list[str]:
        return [self.first_parent, self.second_parent, self.child]

    @property
    def order_key(self) -> str:
        return _order_key(
            self.first_parent,
            self.second_parent,
            self.child,
        )


def _fit_explicit_dvine_order(
    u_df: pd.DataFrame,
    child: str,
    first_parent: str,
    second_parent: str,
    controls: Any,
    clip_eps: float = 1e-6,
) -> FittedExplicitDvineOrder:
    """Fit the explicit D-vine ``first_parent--second_parent--child``."""
    required = {first_parent, second_parent, child}
    missing = required.difference(u_df.columns)
    if missing:
        raise KeyError(
            "Missing variables required for D-vine fitting: "
            + ", ".join(sorted(missing))
        )

    if len(u_df) < 3:
        raise ValueError("At least three observations are required for fitting.")

    u_first = _clip_u(
        u_df[first_parent].to_numpy(dtype=float),
        eps=clip_eps,
    )
    u_second = _clip_u(
        u_df[second_parent].to_numpy(dtype=float),
        eps=clip_eps,
    )
    u_child = _clip_u(
        u_df[child].to_numpy(dtype=float),
        eps=clip_eps,
    )

    data_first_second = np.asfortranarray(
        np.column_stack([u_first, u_second]),
        dtype=float,
    )
    data_second_child = np.asfortranarray(
        np.column_stack([u_second, u_child]),
        dtype=float,
    )

    c_first_second = _fit_bicop(data_first_second, controls=controls)
    c_second_child = _fit_bicop(data_second_child, controls=controls)

    # F(first_parent | second_parent)
    w_first = c_first_second.hfunc2(data_first_second)

    # F(child | second_parent)
    w_child = c_second_child.hfunc1(data_second_child)

    w_first = _clip_u(np.asarray(w_first, dtype=float), eps=clip_eps)
    w_child = _clip_u(np.asarray(w_child, dtype=float), eps=clip_eps)

    data_first_child_given_second = np.asfortranarray(
        np.column_stack([w_first, w_child]),
        dtype=float,
    )
    c_first_child_given_second = _fit_bicop(
        data_first_child_given_second,
        controls=controls,
    )

    loglik = (
        _model_loglik(c_first_second, data_first_second)
        + _model_loglik(c_second_child, data_second_child)
        + _model_loglik(
            c_first_child_given_second,
            data_first_child_given_second,
        )
    )
    npars = (
        _model_npars(c_first_second)
        + _model_npars(c_second_child)
        + _model_npars(c_first_child_given_second)
    )
    n_obs = len(u_df)
    aic = -2.0 * loglik + 2.0 * npars
    bic = -2.0 * loglik + np.log(n_obs) * npars

    return FittedExplicitDvineOrder(
        child=child,
        first_parent=first_parent,
        second_parent=second_parent,
        c_first_second=c_first_second,
        c_second_child=c_second_child,
        c_first_child_given_second=c_first_child_given_second,
        data_first_second=data_first_second,
        data_second_child=data_second_child,
        data_first_child_given_second=data_first_child_given_second,
        n_obs=n_obs,
        npars=float(npars),
        loglik=float(loglik),
        aic=float(aic),
        bic=float(bic),
    )


def conditional_loglik_values_for_order(
    fitted_order: FittedExplicitDvineOrder,
    validation_df: pd.DataFrame,
    clip_eps: float = 1e-6,
    density_floor: float = 1e-300,
) -> np.ndarray:
    """Return observation-wise log ``c(child | first_parent, second_parent)``.

    For the order ``A--B--X`` the conditional copula density is

    ``c_BX(u_B, u_X) * c_AX|B(u_A|B, u_X|B)``.

    The factor ``c_AB`` is deliberately omitted because it belongs to the
    parent density and cancels in the conditional likelihood.
    """
    if validation_df.empty:
        raise ValueError("Validation data must not be empty.")

    first_parent = fitted_order.first_parent
    second_parent = fitted_order.second_parent
    child = fitted_order.child

    u_first = _clip_u(
        validation_df[first_parent].to_numpy(dtype=float),
        eps=clip_eps,
    )
    u_second = _clip_u(
        validation_df[second_parent].to_numpy(dtype=float),
        eps=clip_eps,
    )
    u_child = _clip_u(
        validation_df[child].to_numpy(dtype=float),
        eps=clip_eps,
    )

    data_first_second = np.asfortranarray(
        np.column_stack([u_first, u_second]),
        dtype=float,
    )
    data_second_child = np.asfortranarray(
        np.column_stack([u_second, u_child]),
        dtype=float,
    )

    w_first = fitted_order.c_first_second.hfunc2(data_first_second)
    w_child = fitted_order.c_second_child.hfunc1(data_second_child)

    w_first = _clip_u(np.asarray(w_first, dtype=float), eps=clip_eps)
    w_child = _clip_u(np.asarray(w_child, dtype=float), eps=clip_eps)

    conditional_data = np.asfortranarray(
        np.column_stack([w_first, w_child]),
        dtype=float,
    )

    density_second_child = np.asarray(
        fitted_order.c_second_child.pdf(data_second_child),
        dtype=float,
    ).reshape(-1)
    density_conditional = np.asarray(
        fitted_order.c_first_child_given_second.pdf(conditional_data),
        dtype=float,
    ).reshape(-1)

    density_second_child = np.clip(
        density_second_child,
        density_floor,
        None,
    )
    density_conditional = np.clip(
        density_conditional,
        density_floor,
        None,
    )

    values = np.log(density_second_child) + np.log(density_conditional)

    if not np.isfinite(values).all():
        raise RuntimeError(
            "Conditional log-likelihood contains non-finite values."
        )

    return values


def conditional_mean_loglik_for_order(
    fitted_order: FittedExplicitDvineOrder,
    validation_df: pd.DataFrame,
    clip_eps: float = 1e-6,
) -> float:
    values = conditional_loglik_values_for_order(
        fitted_order=fitted_order,
        validation_df=validation_df,
        clip_eps=clip_eps,
    )
    return float(np.mean(values))


def _joint_loglik_values_for_order(
    fitted_order: FittedExplicitDvineOrder,
    data_df: pd.DataFrame,
    clip_eps: float = 1e-6,
    density_floor: float = 1e-300,
) -> np.ndarray:
    """Return observation-wise log density of the full 3D fitted D-vine."""
    first_parent = fitted_order.first_parent
    second_parent = fitted_order.second_parent
    child = fitted_order.child

    u_first = _clip_u(data_df[first_parent].to_numpy(dtype=float), clip_eps)
    u_second = _clip_u(data_df[second_parent].to_numpy(dtype=float), clip_eps)
    u_child = _clip_u(data_df[child].to_numpy(dtype=float), clip_eps)

    data_first_second = np.asfortranarray(
        np.column_stack([u_first, u_second]), dtype=float
    )
    data_second_child = np.asfortranarray(
        np.column_stack([u_second, u_child]), dtype=float
    )

    w_first = _clip_u(
        fitted_order.c_first_second.hfunc2(data_first_second),
        eps=clip_eps,
    )
    w_child = _clip_u(
        fitted_order.c_second_child.hfunc1(data_second_child),
        eps=clip_eps,
    )
    conditional_data = np.asfortranarray(
        np.column_stack([w_first, w_child]), dtype=float
    )

    densities = [
        fitted_order.c_first_second.pdf(data_first_second),
        fitted_order.c_second_child.pdf(data_second_child),
        fitted_order.c_first_child_given_second.pdf(conditional_data),
    ]

    log_density = np.zeros(len(data_df), dtype=float)
    for density in densities:
        density_array = np.asarray(density, dtype=float).reshape(-1)
        log_density += np.log(np.clip(density_array, density_floor, None))

    if not np.isfinite(log_density).all():
        raise RuntimeError("Joint log-likelihood contains non-finite values.")

    return log_density


# ---------------------------------------------------------------------------
# Serializable library records
# ---------------------------------------------------------------------------


@dataclass
class ConditionalVineRecord:
    dataset_id: str
    child: str
    parent_1: str
    parent_2: str
    parent_set: list[str]
    selected_order: list[str]
    variables: list[str]
    parent_set_size: int
    n_obs: int
    n_variables: int

    selection_criterion: str
    candidate_families: list[str]

    order_strategy_requested: str
    order_selection_method: str
    order_selection_metric: str
    order_validation_fraction: float | None
    order_selection_seed: int | None
    order_training_size: int | None
    order_validation_size: int | None
    minimum_validation_rows: int
    candidate_order_scores: dict[str, float | None]
    candidate_order_statuses: dict[str, str]
    candidate_order_errors: dict[str, str | None]
    selected_order_score: float | None
    alternative_order_score: float | None
    order_score_margin: float | None

    conditional_model_strategy_requested: str
    conditional_model_type: str
    simplifying_assumption_selected: bool
    selected_number_bins: int
    conditional_family: str | None
    conditional_rotation: int | None
    simplified_validation_score: float | None
    selected_validation_score: float | None
    score_improvement_over_simplified: float | None
    conditional_bin_edges: list[float]
    conditional_bin_counts: list[int]
    conditional_bin_model_files: list[str]
    conditional_tau_diagnostics: list[dict[str, Any]]
    conditional_model_metadata: dict[str, Any]

    model_file: str
    model_format: str

    bicop_parent_1_parent_2_file: str
    bicop_parent_2_child_file: str
    bicop_parent_1_child_given_parent_2_file: str

    family_parent_1_parent_2: str | None
    family_parent_2_child: str | None
    family_parent_1_child_given_parent_2: str | None

    rotation_parent_1_parent_2: int | None
    rotation_parent_2_child: int | None
    rotation_parent_1_child_given_parent_2: int | None

    parameters_parent_1_parent_2: list
    parameters_parent_2_child: list
    parameters_parent_1_child_given_parent_2: list

    npars: float | None
    loglik: float | None
    aic: float | None
    bic: float | None

    kendall_parent_1_child: float | None
    kendall_parent_2_child: float | None
    kendall_parent_1_parent_2: float | None

    spearman_parent_1_child: float | None
    spearman_parent_2_child: float | None
    spearman_parent_1_parent_2: float | None

    pair_family_summary: str | None
    vine_structure: str | None

    status: str
    error: str | None = None


@dataclass
class ConditionalVineLibrary:
    dataset_id: str
    columns: list[str]
    n_rows: int
    parent_set_size: int
    selection_criterion: str
    family_set: list[str]
    order_strategy: str
    order_validation_fraction: float
    order_selection_seed: int
    minimum_validation_rows: int
    conditional_model_strategy: str
    conditional_candidate_bin_counts: list[int]
    conditional_minimum_bin_size: int
    conditional_validation_fraction: float
    conditional_selection_seed: int
    conditional_minimum_score_improvement: float
    records: list[ConditionalVineRecord]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(record) for record in self.records])

    def save_pickle(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "dataset_id": self.dataset_id,
            "columns": self.columns,
            "n_rows": self.n_rows,
            "parent_set_size": self.parent_set_size,
            "selection_criterion": self.selection_criterion,
            "family_set": self.family_set,
            "order_strategy": self.order_strategy,
            "order_validation_fraction": self.order_validation_fraction,
            "order_selection_seed": self.order_selection_seed,
            "minimum_validation_rows": self.minimum_validation_rows,
            "conditional_model_strategy": self.conditional_model_strategy,
            "conditional_candidate_bin_counts": self.conditional_candidate_bin_counts,
            "conditional_minimum_bin_size": self.conditional_minimum_bin_size,
            "conditional_validation_fraction": self.conditional_validation_fraction,
            "conditional_selection_seed": self.conditional_selection_seed,
            "conditional_minimum_score_improvement": (
                self.conditional_minimum_score_improvement
            ),
            "records": [asdict(record) for record in self.records],
            "implementation": (
                "explicit_3d_dvine_bicop_hfunctions_flexible_parent_order_v1"
            ),
            "conditional_implementation": (
                "adaptive_quantile_binned_conditional_copula_v1"
            ),
        }

        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @staticmethod
    def load_pickle(path: str | Path) -> dict:
        path = Path(path)
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# Order selection and final fitting
# ---------------------------------------------------------------------------


def _make_train_validation_split(
    u_df: pd.DataFrame,
    validation_fraction: float,
    seed: int,
    minimum_validation_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            "order_validation_fraction must lie strictly between 0 and 1."
        )
    if minimum_validation_rows < 1:
        raise ValueError("minimum_validation_rows must be positive.")

    n_obs = len(u_df)
    validation_size = int(np.floor(validation_fraction * n_obs))
    training_size = n_obs - validation_size

    # A held-out comparison is only used when both partitions are sufficiently
    # stable. Otherwise the implementation falls back to full-sample BIC.
    minimum_training_rows = max(10, minimum_validation_rows)
    if (
        validation_size < minimum_validation_rows
        or training_size < minimum_training_rows
    ):
        return None

    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(n_obs)
    validation_indices = permutation[:validation_size]
    training_indices = permutation[validation_size:]

    training_df = u_df.iloc[training_indices].reset_index(drop=True)
    validation_df = u_df.iloc[validation_indices].reset_index(drop=True)
    return training_df, validation_df


def _select_order(
    u_df: pd.DataFrame,
    child: str,
    parent_1: str,
    parent_2: str,
    controls: Any,
    order_strategy: OrderStrategy,
    order_validation_fraction: float,
    order_selection_seed: int,
    minimum_validation_rows: int,
    clip_eps: float,
) -> tuple[
    tuple[str, str],
    str,
    str,
    dict[str, float | None],
    dict[str, str],
    dict[str, str | None],
    float | None,
    float | None,
    float | None,
    int | None,
    int | None,
]:
    """Select the parent order and return complete selection diagnostics."""
    canonical_first, canonical_second = _canonical_parent_pair(
        parent_1,
        parent_2,
    )
    candidate_orders = [
        (canonical_first, canonical_second),
        (canonical_second, canonical_first),
    ]

    candidate_scores: dict[str, float | None] = {
        _order_key(a, b, child): None for a, b in candidate_orders
    }
    candidate_statuses: dict[str, str] = {
        _order_key(a, b, child): "not_evaluated" for a, b in candidate_orders
    }
    candidate_errors: dict[str, str | None] = {
        _order_key(a, b, child): None for a, b in candidate_orders
    }

    if order_strategy == "fixed":
        # Fixed mode preserves the supplied order for explicit backward
        # compatibility. It does not compare the alternative.
        selected = (str(parent_1), str(parent_2))
        selected_key = _order_key(*selected, child)
        candidate_statuses[selected_key] = "selected_without_comparison"
        return (
            selected,
            "fixed_parent_order",
            "not_applicable",
            candidate_scores,
            candidate_statuses,
            candidate_errors,
            None,
            None,
            None,
            None,
            None,
        )

    split = None
    if order_strategy == "heldout_conditional_loglik":
        split = _make_train_validation_split(
            u_df=u_df,
            validation_fraction=order_validation_fraction,
            seed=order_selection_seed,
            minimum_validation_rows=minimum_validation_rows,
        )

    if split is not None:
        fit_df, scoring_df = split
        method = "heldout_conditional_loglik"
        metric = "mean_conditional_loglik"
        training_size = len(fit_df)
        validation_size = len(scoring_df)
        score_function = lambda model: conditional_mean_loglik_for_order(
            fitted_order=model,
            validation_df=scoring_df,
            clip_eps=clip_eps,
        )
    else:
        # This is also the direct path when order_strategy == "bic".
        fit_df = u_df
        method = (
            "bic"
            if order_strategy == "bic"
            else "bic_fallback_insufficient_validation_rows"
        )
        metric = "negative_bic"
        training_size = len(u_df)
        validation_size = None
        score_function = lambda model: -float(model.bic)

    successful: list[tuple[tuple[str, str], float]] = []

    for first_parent, second_parent in candidate_orders:
        key = _order_key(first_parent, second_parent, child)
        try:
            candidate = _fit_explicit_dvine_order(
                u_df=fit_df,
                child=child,
                first_parent=first_parent,
                second_parent=second_parent,
                controls=controls,
                clip_eps=clip_eps,
            )
            score = float(score_function(candidate))
            if not np.isfinite(score):
                raise RuntimeError("The candidate order score is not finite.")
            candidate_scores[key] = score
            candidate_statuses[key] = "ok"
            successful.append(((first_parent, second_parent), score))
        except Exception as exc:
            candidate_statuses[key] = "failed"
            candidate_errors[key] = f"{type(exc).__name__}: {exc}"

    if not successful:
        raise RuntimeError(
            "Both candidate D-vine parent orders failed. "
            f"Errors: {candidate_errors}"
        )

    # Higher scores are always better. In BIC mode the score is negative BIC.
    # The secondary lexical key makes ties deterministic and independent of
    # the order in which the parent names were supplied.
    successful.sort(
        key=lambda item: (
            -item[1],
            item[0][0],
            item[0][1],
        )
    )
    selected_order, selected_score = successful[0]

    alternative_score = successful[1][1] if len(successful) > 1 else None
    margin = (
        float(selected_score - alternative_score)
        if alternative_score is not None
        else None
    )

    return (
        selected_order,
        method,
        metric,
        candidate_scores,
        candidate_statuses,
        candidate_errors,
        float(selected_score),
        float(alternative_score) if alternative_score is not None else None,
        margin,
        training_size,
        validation_size,
    )


def _empty_failed_record(
    *,
    dataset_id: str,
    child: str,
    parent_1: str,
    parent_2: str,
    parent_set: list[str],
    selected_order: list[str],
    n_obs: int,
    selection_criterion: str,
    candidate_families: list[str],
    order_strategy_requested: str,
    order_selection_method: str,
    order_selection_metric: str,
    order_validation_fraction: float | None,
    order_selection_seed: int | None,
    order_training_size: int | None,
    order_validation_size: int | None,
    minimum_validation_rows: int,
    candidate_order_scores: dict[str, float | None],
    candidate_order_statuses: dict[str, str],
    candidate_order_errors: dict[str, str | None],
    selected_order_score: float | None,
    alternative_order_score: float | None,
    order_score_margin: float | None,
    model_paths: tuple[Path, Path, Path],
    error: str,
) -> ConditionalVineRecord:
    c12_path, c23_path, c13_path = model_paths
    return ConditionalVineRecord(
        dataset_id=dataset_id,
        child=child,
        parent_1=parent_1,
        parent_2=parent_2,
        parent_set=parent_set,
        selected_order=selected_order,
        variables=selected_order,
        parent_set_size=2,
        n_obs=n_obs,
        n_variables=3,
        selection_criterion=selection_criterion,
        candidate_families=candidate_families,
        order_strategy_requested=order_strategy_requested,
        order_selection_method=order_selection_method,
        order_selection_metric=order_selection_metric,
        order_validation_fraction=order_validation_fraction,
        order_selection_seed=order_selection_seed,
        order_training_size=order_training_size,
        order_validation_size=order_validation_size,
        minimum_validation_rows=minimum_validation_rows,
        candidate_order_scores=candidate_order_scores,
        candidate_order_statuses=candidate_order_statuses,
        candidate_order_errors=candidate_order_errors,
        selected_order_score=selected_order_score,
        alternative_order_score=alternative_order_score,
        order_score_margin=order_score_margin,
        conditional_model_strategy_requested="not_completed",
        conditional_model_type="failed",
        simplifying_assumption_selected=True,
        selected_number_bins=1,
        conditional_family=None,
        conditional_rotation=None,
        simplified_validation_score=None,
        selected_validation_score=None,
        score_improvement_over_simplified=None,
        conditional_bin_edges=[],
        conditional_bin_counts=[],
        conditional_bin_model_files=[],
        conditional_tau_diagnostics=[],
        conditional_model_metadata={},
        model_file=str(c13_path),
        model_format="explicit_3d_dvine_bicop_json",
        bicop_parent_1_parent_2_file=str(c12_path),
        bicop_parent_2_child_file=str(c23_path),
        bicop_parent_1_child_given_parent_2_file=str(c13_path),
        family_parent_1_parent_2=None,
        family_parent_2_child=None,
        family_parent_1_child_given_parent_2=None,
        rotation_parent_1_parent_2=None,
        rotation_parent_2_child=None,
        rotation_parent_1_child_given_parent_2=None,
        parameters_parent_1_parent_2=[],
        parameters_parent_2_child=[],
        parameters_parent_1_child_given_parent_2=[],
        npars=None,
        loglik=None,
        aic=None,
        bic=None,
        kendall_parent_1_child=None,
        kendall_parent_2_child=None,
        kendall_parent_1_parent_2=None,
        spearman_parent_1_child=None,
        spearman_parent_2_child=None,
        spearman_parent_1_parent_2=None,
        pair_family_summary=None,
        vine_structure=None,
        status="failed",
        error=error,
    )


def fit_one_conditional_vine(
    u_df: pd.DataFrame,
    child: str,
    parent_1: str,
    parent_2: str,
    dataset_id: str,
    controls: Any,
    candidate_families: list[str],
    selection_criterion: str,
    model_dir: str | Path,
    clip_eps: float = 1e-6,
    order_strategy: str = "heldout_conditional_loglik",
    order_validation_fraction: float = 0.20,
    order_selection_seed: int = 42,
    minimum_validation_rows: int = 50,
    conditional_model_strategy: str = "adaptive_quantile_bins",
    conditional_candidate_bin_counts: Sequence[int] = (1, 2, 3, 4, 5, 6),
    conditional_minimum_bin_size: int = 75,
    conditional_validation_fraction: float = 0.20,
    conditional_selection_seed: int = 42,
    conditional_minimum_score_improvement: float = 0.005,
) -> ConditionalVineRecord:
    """Fit a flexible-order two-parent mechanism with optional adaptive bins.

    The supplied DAG parent set is never changed.  First, the two admissible
    D-vine parent orders are compared.  For the selected order ``A--B--X``,
    the method can then compare the simplifying assumption (one global
    ``C(A,X|B)``) against equal-frequency, piecewise-constant conditional
    copulas indexed by ``U_B``.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    strategy = _validate_order_strategy(order_strategy)
    conditional_strategy = _validate_conditional_model_strategy(
        conditional_model_strategy
    )
    conditional_candidates = normalize_candidate_bin_counts(
        conditional_candidate_bin_counts
    )
    if conditional_minimum_bin_size < 1:
        raise ValueError("conditional_minimum_bin_size must be positive.")
    if not 0.0 < conditional_validation_fraction < 1.0:
        raise ValueError(
            "conditional_validation_fraction must lie strictly between 0 and 1."
        )
    if conditional_minimum_score_improvement < 0.0:
        raise ValueError(
            "conditional_minimum_score_improvement must be non-negative."
        )

    canonical_parent_1, canonical_parent_2 = _canonical_parent_pair(
        parent_1,
        parent_2,
    )
    parent_set = [canonical_parent_1, canonical_parent_2]

    selected_parent_1 = canonical_parent_1
    selected_parent_2 = canonical_parent_2
    selection_method = "not_completed"
    selection_metric = "not_completed"
    candidate_scores: dict[str, float | None] = {}
    candidate_statuses: dict[str, str] = {}
    candidate_errors: dict[str, str | None] = {}
    selected_score: float | None = None
    alternative_score: float | None = None
    score_margin: float | None = None
    training_size: int | None = None
    validation_size: int | None = None

    placeholder_name = (
        f"child_{_safe_filename(child)}"
        f"__given__{_safe_filename(selected_parent_1)}"
        f"__{_safe_filename(selected_parent_2)}"
    )
    placeholder_paths = (
        model_dir / f"{placeholder_name}__c_parent1_parent2.json",
        model_dir / f"{placeholder_name}__c_parent2_child.json",
        model_dir / f"{placeholder_name}__c_parent1_child_given_parent2.json",
    )

    try:
        (
            selected_pair,
            selection_method,
            selection_metric,
            candidate_scores,
            candidate_statuses,
            candidate_errors,
            selected_score,
            alternative_score,
            score_margin,
            training_size,
            validation_size,
        ) = _select_order(
            u_df=u_df,
            child=child,
            parent_1=parent_1,
            parent_2=parent_2,
            controls=controls,
            order_strategy=strategy,
            order_validation_fraction=order_validation_fraction,
            order_selection_seed=order_selection_seed,
            minimum_validation_rows=minimum_validation_rows,
            clip_eps=clip_eps,
        )

        selected_parent_1, selected_parent_2 = selected_pair
        selected_order = [selected_parent_1, selected_parent_2, child]

        conditional_selection = None
        conditional_metadata: dict[str, Any] = {
            "selection_method": "simplified_requested",
            "candidate_bin_counts": conditional_candidates,
            "candidate_validation_scores": {"1": None},
            "candidate_statuses": {"1": "selected_without_comparison"},
            "candidate_errors": {"1": None},
            "minimum_bin_size": int(conditional_minimum_bin_size),
            "validation_fraction": float(conditional_validation_fraction),
            "selection_seed": int(conditional_selection_seed),
            "minimum_score_improvement": float(
                conditional_minimum_score_improvement
            ),
            "fixed_family_across_bins": True,
            "training_size": None,
            "validation_size": None,
            "raw_best_number_bins": 1,
            "raw_best_validation_score": None,
        }

        if conditional_strategy == "adaptive_quantile_bins":
            conditional_split = _make_train_validation_split(
                u_df=u_df,
                validation_fraction=conditional_validation_fraction,
                seed=conditional_selection_seed,
                minimum_validation_rows=max(
                    20,
                    min(50, int(conditional_minimum_bin_size)),
                ),
            )
            if conditional_split is not None:
                conditional_training_df, conditional_validation_df = (
                    conditional_split
                )
                fitted_training_order = _fit_explicit_dvine_order(
                    u_df=conditional_training_df,
                    child=child,
                    first_parent=selected_parent_1,
                    second_parent=selected_parent_2,
                    controls=controls,
                    clip_eps=clip_eps,
                )
                conditional_selection = select_number_of_quantile_bins(
                    fitted_training_order=fitted_training_order,
                    training_df=conditional_training_df,
                    validation_df=conditional_validation_df,
                    candidate_bin_counts=conditional_candidates,
                    minimum_bin_size=conditional_minimum_bin_size,
                    minimum_score_improvement=(
                        conditional_minimum_score_improvement
                    ),
                    clip_eps=clip_eps,
                )
                conditional_metadata = {
                    "selection_method": conditional_selection.method,
                    "candidate_bin_counts": conditional_candidates,
                    "candidate_validation_scores": (
                        conditional_selection.candidate_scores
                    ),
                    "candidate_statuses": (
                        conditional_selection.candidate_statuses
                    ),
                    "candidate_errors": conditional_selection.candidate_errors,
                    "minimum_bin_size": int(conditional_minimum_bin_size),
                    "validation_fraction": float(
                        conditional_validation_fraction
                    ),
                    "selection_seed": int(conditional_selection_seed),
                    "minimum_score_improvement": float(
                        conditional_minimum_score_improvement
                    ),
                    "fixed_family_across_bins": True,
                    "training_size": conditional_selection.training_size,
                    "validation_size": conditional_selection.validation_size,
                    "raw_best_number_bins": (
                        conditional_selection.raw_best_number_bins
                    ),
                    "raw_best_validation_score": (
                        conditional_selection.raw_best_score
                    ),
                }
            else:
                conditional_metadata.update(
                    {
                        "selection_method": (
                            "simplified_fallback_insufficient_validation_rows"
                        ),
                        "candidate_statuses": {
                            str(value): (
                                "selected_fallback" if value == 1 else "not_evaluated"
                            )
                            for value in conditional_candidates
                        },
                        "candidate_errors": {
                            str(value): None for value in conditional_candidates
                        },
                        "candidate_validation_scores": {
                            str(value): None for value in conditional_candidates
                        },
                    }
                )

        # First-tree models are always refitted on all observations after
        # model selection.
        final_fit = _fit_explicit_dvine_order(
            u_df=u_df,
            child=child,
            first_parent=selected_parent_1,
            second_parent=selected_parent_2,
            controls=controls,
            clip_eps=clip_eps,
        )

        full_conditioning, full_w_first, full_w_child = (
            compute_conditional_coordinates(
                c_first_second=final_fit.c_first_second,
                c_second_child=final_fit.c_second_child,
                data=u_df,
                first_parent=selected_parent_1,
                second_parent=selected_parent_2,
                child=child,
                clip_eps=clip_eps,
            )
        )

        if conditional_selection is None:
            selected_number_bins = 1
            conditional_family_enum = final_fit.c_first_child_given_second.family
            conditional_family = _enum_to_string(conditional_family_enum)
            conditional_rotation = int(
                final_fit.c_first_child_given_second.rotation
            )
            simplified_validation_score = None
            selected_validation_score = None
            score_improvement = None
        else:
            selected_number_bins = conditional_selection.selected_number_bins
            conditional_family_enum = conditional_selection.family_enum
            conditional_family = conditional_selection.family
            conditional_rotation = conditional_selection.rotation
            simplified_validation_score = conditional_selection.simplified_score
            selected_validation_score = conditional_selection.selected_score
            score_improvement = (
                conditional_selection.score_improvement_over_simplified
            )

        # Store a global model for backward compatibility and for direct
        # comparison with the simplified model, even when bins are selected.
        global_conditional_model = fit_fixed_family_rotation_bicop(
            np.column_stack([full_w_first, full_w_child]),
            family=conditional_family_enum,
            rotation=conditional_rotation,
            clip_eps=clip_eps,
        )

        if selected_number_bins == 1:
            selected_conditional_model = FittedBinnedConditionalCopula(
                family=conditional_family,
                rotation=conditional_rotation,
                bin_edges=np.asarray([0.0, 1.0], dtype=float),
                bin_counts=[len(u_df)],
                models=[global_conditional_model],
            )
        else:
            try:
                selected_conditional_model = fit_binned_conditional_copula(
                    w_first=full_w_first,
                    w_child=full_w_child,
                    conditioning=full_conditioning,
                    family=conditional_family_enum,
                    rotation=conditional_rotation,
                    n_bins=selected_number_bins,
                    minimum_bin_size=conditional_minimum_bin_size,
                    clip_eps=clip_eps,
                )
            except Exception as refit_error:
                # A rare full-data refit failure must not make the complete
                # local mechanism unusable. Fall back transparently to the
                # simplified model and preserve the reason in metadata.
                conditional_metadata["full_data_refit_fallback_error"] = (
                    f"{type(refit_error).__name__}: {refit_error}"
                )
                selected_number_bins = 1
                selected_validation_score = simplified_validation_score
                selected_conditional_model = FittedBinnedConditionalCopula(
                    family=conditional_family,
                    rotation=conditional_rotation,
                    bin_edges=np.asarray([0.0, 1.0], dtype=float),
                    bin_counts=[len(u_df)],
                    models=[global_conditional_model],
                )

        conditional_model_type = (
            "simplified" if selected_number_bins == 1 else "quantile_binned"
        )
        simplifying_assumption_selected = selected_number_bins == 1

        model_name = (
            f"child_{_safe_filename(child)}"
            f"__given__{_safe_filename(selected_parent_1)}"
            f"__{_safe_filename(selected_parent_2)}"
        )
        c12_path = model_dir / f"{model_name}__c_parent1_parent2.json"
        c23_path = model_dir / f"{model_name}__c_parent2_child.json"
        c13_path = (
            model_dir / f"{model_name}__c_parent1_child_given_parent2.json"
        )

        c12_file = _save_bicop(final_fit.c_first_second, c12_path)
        c23_file = _save_bicop(final_fit.c_second_child, c23_path)
        c13_file = _save_bicop(global_conditional_model, c13_path)

        conditional_bin_model_files: list[str] = []
        if selected_number_bins == 1:
            conditional_bin_model_files = [c13_file]
        else:
            for bin_index, model in enumerate(selected_conditional_model.models):
                path = model_dir / (
                    f"{model_name}__conditional_bin_{bin_index:02d}.json"
                )
                conditional_bin_model_files.append(_save_bicop(model, path))

        s12 = _bicop_summary(final_fit.c_first_second)
        s23 = _bicop_summary(final_fit.c_second_child)
        s13 = _bicop_summary(global_conditional_model)

        conditional_loglik = float(
            np.sum(
                selected_conditional_model.loglik_values(
                    w_first=full_w_first,
                    w_child=full_w_child,
                    conditioning=full_conditioning,
                    clip_eps=clip_eps,
                )
            )
        )
        loglik = (
            _model_loglik(final_fit.c_first_second, final_fit.data_first_second)
            + _model_loglik(final_fit.c_second_child, final_fit.data_second_child)
            + conditional_loglik
        )
        npars = (
            _model_npars(final_fit.c_first_second)
            + _model_npars(final_fit.c_second_child)
            + sum(_model_npars(model) for model in selected_conditional_model.models)
        )
        aic = -2.0 * loglik + 2.0 * npars
        bic = -2.0 * loglik + np.log(len(u_df)) * npars

        diagnostics = selected_bin_diagnostics(
            model=selected_conditional_model,
            w_first=full_w_first,
            w_child=full_w_child,
            conditioning=full_conditioning,
        )
        conditional_metadata.update(
            {
                "selected_number_bins": int(selected_number_bins),
                "conditional_family": conditional_family,
                "conditional_rotation": int(conditional_rotation),
                "bin_model_summaries": selected_conditional_model.summaries(),
            }
        )

        u1 = _clip_u(
            u_df[selected_parent_1].to_numpy(dtype=float), eps=clip_eps
        )
        u2 = _clip_u(
            u_df[selected_parent_2].to_numpy(dtype=float), eps=clip_eps
        )
        uy = _clip_u(u_df[child].to_numpy(dtype=float), eps=clip_eps)

        pair_family_summary = (
            f"C({selected_parent_1},{selected_parent_2})={s12['family']}; "
            f"C({selected_parent_2},{child})={s23['family']}; "
            f"C({selected_parent_1},{child}|{selected_parent_2})="
            f"{conditional_family} in {selected_number_bins} bin(s)"
        )

        return ConditionalVineRecord(
            dataset_id=dataset_id,
            child=child,
            parent_1=selected_parent_1,
            parent_2=selected_parent_2,
            parent_set=parent_set,
            selected_order=selected_order,
            variables=selected_order,
            parent_set_size=2,
            n_obs=len(u_df),
            n_variables=3,
            selection_criterion=selection_criterion,
            candidate_families=candidate_families,
            order_strategy_requested=strategy,
            order_selection_method=selection_method,
            order_selection_metric=selection_metric,
            order_validation_fraction=(
                float(order_validation_fraction)
                if selection_method.startswith("heldout")
                else None
            ),
            order_selection_seed=(
                int(order_selection_seed)
                if selection_method.startswith("heldout")
                else None
            ),
            order_training_size=training_size,
            order_validation_size=validation_size,
            minimum_validation_rows=int(minimum_validation_rows),
            candidate_order_scores=candidate_scores,
            candidate_order_statuses=candidate_statuses,
            candidate_order_errors=candidate_errors,
            selected_order_score=selected_score,
            alternative_order_score=alternative_score,
            order_score_margin=score_margin,
            conditional_model_strategy_requested=conditional_strategy,
            conditional_model_type=conditional_model_type,
            simplifying_assumption_selected=simplifying_assumption_selected,
            selected_number_bins=int(selected_number_bins),
            conditional_family=conditional_family,
            conditional_rotation=int(conditional_rotation),
            simplified_validation_score=simplified_validation_score,
            selected_validation_score=selected_validation_score,
            score_improvement_over_simplified=score_improvement,
            conditional_bin_edges=(
                np.asarray(selected_conditional_model.bin_edges, dtype=float)
                .reshape(-1)
                .tolist()
            ),
            conditional_bin_counts=[
                int(value) for value in selected_conditional_model.bin_counts
            ],
            conditional_bin_model_files=conditional_bin_model_files,
            conditional_tau_diagnostics=diagnostics,
            conditional_model_metadata=conditional_metadata,
            model_file=c13_file,
            model_format=(
                "explicit_3d_dvine_quantile_binned_bicop_json"
                if selected_number_bins > 1
                else "explicit_3d_dvine_bicop_json"
            ),
            bicop_parent_1_parent_2_file=c12_file,
            bicop_parent_2_child_file=c23_file,
            bicop_parent_1_child_given_parent_2_file=c13_file,
            family_parent_1_parent_2=s12["family"],
            family_parent_2_child=s23["family"],
            family_parent_1_child_given_parent_2=s13["family"],
            rotation_parent_1_parent_2=s12["rotation"],
            rotation_parent_2_child=s23["rotation"],
            rotation_parent_1_child_given_parent_2=s13["rotation"],
            parameters_parent_1_parent_2=s12["parameters"],
            parameters_parent_2_child=s23["parameters"],
            parameters_parent_1_child_given_parent_2=s13["parameters"],
            npars=float(npars),
            loglik=float(loglik),
            aic=float(aic),
            bic=float(bic),
            kendall_parent_1_child=_safe_float(kendalltau(u1, uy)[0]),
            kendall_parent_2_child=_safe_float(kendalltau(u2, uy)[0]),
            kendall_parent_1_parent_2=_safe_float(kendalltau(u1, u2)[0]),
            spearman_parent_1_child=_safe_float(spearmanr(u1, uy)[0]),
            spearman_parent_2_child=_safe_float(spearmanr(u2, uy)[0]),
            spearman_parent_1_parent_2=_safe_float(spearmanr(u1, u2)[0]),
            pair_family_summary=pair_family_summary,
            vine_structure=(
                f"explicit D-vine: {selected_parent_1} -- "
                f"{selected_parent_2} -- {child}; conditional model="
                f"{conditional_model_type} ({selected_number_bins} bin(s))"
            ),
            status="ok",
            error=None,
        )

    except Exception as exc:
        selected_order = [selected_parent_1, selected_parent_2, child]
        model_name = (
            f"child_{_safe_filename(child)}"
            f"__given__{_safe_filename(selected_parent_1)}"
            f"__{_safe_filename(selected_parent_2)}"
        )
        failed_paths = (
            model_dir / f"{model_name}__c_parent1_parent2.json",
            model_dir / f"{model_name}__c_parent2_child.json",
            model_dir / f"{model_name}__c_parent1_child_given_parent2.json",
        )
        record = _empty_failed_record(
            dataset_id=dataset_id,
            child=child,
            parent_1=selected_parent_1,
            parent_2=selected_parent_2,
            parent_set=parent_set,
            selected_order=selected_order,
            n_obs=len(u_df),
            selection_criterion=selection_criterion,
            candidate_families=candidate_families,
            order_strategy_requested=strategy,
            order_selection_method=selection_method,
            order_selection_metric=selection_metric,
            order_validation_fraction=(
                float(order_validation_fraction)
                if selection_method.startswith("heldout")
                else None
            ),
            order_selection_seed=(
                int(order_selection_seed)
                if selection_method.startswith("heldout")
                else None
            ),
            order_training_size=training_size,
            order_validation_size=validation_size,
            minimum_validation_rows=int(minimum_validation_rows),
            candidate_order_scores=candidate_scores,
            candidate_order_statuses=candidate_statuses,
            candidate_order_errors=candidate_errors,
            selected_order_score=selected_score,
            alternative_order_score=alternative_score,
            order_score_margin=score_margin,
            model_paths=failed_paths,
            error=f"{type(exc).__name__}: {exc}",
        )
        record.conditional_model_strategy_requested = conditional_strategy
        record.conditional_model_metadata = {
            "candidate_bin_counts": conditional_candidates,
            "minimum_bin_size": int(conditional_minimum_bin_size),
            "validation_fraction": float(conditional_validation_fraction),
            "selection_seed": int(conditional_selection_seed),
            "minimum_score_improvement": float(
                conditional_minimum_score_improvement
            ),
        }
        return record

def build_conditional_vine_library(
    u_df: pd.DataFrame,
    dataset_id: str,
    parent_set_size: int = 2,
    selection_criterion: str = "bic",
    allow_rotations: bool = True,
    num_threads: int = 1,
    model_dir: str | Path = "artifacts/conditional_vines_indegree2",
    order_strategy: str = "heldout_conditional_loglik",
    order_validation_fraction: float = 0.20,
    order_selection_seed: int = 42,
    minimum_validation_rows: int = 50,
    conditional_model_strategy: str = "adaptive_quantile_bins",
    conditional_candidate_bin_counts: Sequence[int] = (1, 2, 3, 4, 5, 6),
    conditional_minimum_bin_size: int = 75,
    conditional_validation_fraction: float = 0.20,
    conditional_selection_seed: int = 42,
    conditional_minimum_score_improvement: float = 0.005,
    fixed_variable_order: bool | None = None,
    limit: int | None = None,
) -> ConditionalVineLibrary:
    """Fit one flexible-order explicit D-vine for each child/parent-set triple.

    ``fixed_variable_order`` is retained only for compatibility with earlier
    callers. When supplied, ``True`` maps to ``order_strategy='fixed'`` and
    ``False`` maps to ``order_strategy='heldout_conditional_loglik'``.
    """
    if parent_set_size != 2:
        raise NotImplementedError(
            "This builder currently supports parent_set_size=2 only."
        )

    if u_df.empty:
        raise ValueError("Cannot build conditional vine library from empty data.")

    if fixed_variable_order is not None:
        order_strategy = (
            "fixed" if fixed_variable_order else "heldout_conditional_loglik"
        )

    strategy = _validate_order_strategy(order_strategy)
    conditional_strategy = _validate_conditional_model_strategy(
        conditional_model_strategy
    )
    conditional_candidates = normalize_candidate_bin_counts(
        conditional_candidate_bin_counts
    )

    controls, family_set_names = build_bicop_fit_controls(
        selection_criterion=selection_criterion,
        allow_rotations=allow_rotations,
        num_threads=num_threads,
    )

    columns = list(u_df.columns)
    records: list[ConditionalVineRecord] = []

    tasks: list[tuple[str, str, str]] = []
    for child in columns:
        possible_parents = [col for col in columns if col != child]
        for parent_1, parent_2 in combinations(possible_parents, 2):
            tasks.append((child, parent_1, parent_2))

    if limit is not None:
        tasks = tasks[:limit]

    total = len(tasks)

    for idx, (child, parent_1, parent_2) in enumerate(tasks, start=1):
        print(
            f"[{idx:04d}/{total:04d}] "
            f"Fitting flexible-order D-vine: {child} | "
            f"{parent_1}, {parent_2}"
        )

        record = fit_one_conditional_vine(
            u_df=u_df,
            child=child,
            parent_1=parent_1,
            parent_2=parent_2,
            dataset_id=dataset_id,
            controls=controls,
            candidate_families=family_set_names,
            selection_criterion=selection_criterion,
            model_dir=model_dir,
            clip_eps=1e-6,
            order_strategy=strategy,
            order_validation_fraction=order_validation_fraction,
            order_selection_seed=order_selection_seed,
            minimum_validation_rows=minimum_validation_rows,
            conditional_model_strategy=conditional_strategy,
            conditional_candidate_bin_counts=conditional_candidates,
            conditional_minimum_bin_size=conditional_minimum_bin_size,
            conditional_validation_fraction=conditional_validation_fraction,
            conditional_selection_seed=conditional_selection_seed,
            conditional_minimum_score_improvement=(
                conditional_minimum_score_improvement
            ),
        )

        records.append(record)

    return ConditionalVineLibrary(
        dataset_id=dataset_id,
        columns=columns,
        n_rows=len(u_df),
        parent_set_size=parent_set_size,
        selection_criterion=selection_criterion,
        family_set=family_set_names,
        order_strategy=strategy,
        order_validation_fraction=float(order_validation_fraction),
        order_selection_seed=int(order_selection_seed),
        minimum_validation_rows=int(minimum_validation_rows),
        conditional_model_strategy=conditional_strategy,
        conditional_candidate_bin_counts=conditional_candidates,
        conditional_minimum_bin_size=int(conditional_minimum_bin_size),
        conditional_validation_fraction=float(conditional_validation_fraction),
        conditional_selection_seed=int(conditional_selection_seed),
        conditional_minimum_score_improvement=float(
            conditional_minimum_score_improvement
        ),
        records=records,
    )
