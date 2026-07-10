"""Explicit four-dimensional D-vines for three-parent local mechanisms.

For a selected local order ``A -- B -- C -- X`` the model contains six
pair-copulas:

Tree 1
    C(A,B), C(B,C), C(C,X)
Tree 2
    C(A,C | B), C(B,X | C)
Tree 3
    C(A,X | B,C)

The child is always the final variable.  All six permutations of the three
parents can be compared by held-out conditional log-likelihood.  The final
selected order is then refitted on all observations and serialized as six
``pyvinecopulib.Bicop`` JSON files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations, permutations
from pathlib import Path
import pickle
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from copula_causal_sim.copulas.conditional_vine_library import (
    _bicop_summary,
    _clip_u,
    _fit_bicop,
    _model_loglik,
    _model_npars,
    _safe_filename,
    _safe_float,
    _save_bicop,
    build_bicop_fit_controls,
)


OrderStrategy = Literal["heldout_conditional_loglik", "bic", "fixed"]


def _validate_order_strategy(value: str) -> OrderStrategy:
    normalized = str(value).strip().lower()
    allowed = {"heldout_conditional_loglik", "bic", "fixed"}
    if normalized not in allowed:
        raise ValueError(
            f"Unsupported order strategy {value!r}. Expected one of "
            f"{sorted(allowed)}."
        )
    return normalized  # type: ignore[return-value]


def _canonical_parent_set(parents: Sequence[str]) -> tuple[str, str, str]:
    normalized = tuple(sorted(str(value) for value in parents))
    if len(normalized) != 3 or len(set(normalized)) != 3:
        raise ValueError("Exactly three distinct parent variables are required.")
    return normalized  # type: ignore[return-value]


def _order_key(order: Sequence[str], child: str) -> str:
    return "--".join([*map(str, order), str(child)])


def _density_log_values(
    model: Any,
    data: np.ndarray,
    density_floor: float = 1e-300,
) -> np.ndarray:
    density = np.asarray(
        model.pdf(np.asfortranarray(data, dtype=float)),
        dtype=float,
    ).reshape(-1)
    values = np.log(np.clip(density, density_floor, None))
    if not np.isfinite(values).all():
        raise RuntimeError("Copula log-density contains non-finite values.")
    return values


def _make_train_validation_split(
    data: pd.DataFrame,
    validation_fraction: float,
    seed: int,
    minimum_validation_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("validation_fraction must lie strictly between 0 and 1.")
    if int(minimum_validation_rows) < 1:
        raise ValueError("minimum_validation_rows must be positive.")

    n_obs = len(data)
    validation_size = int(np.floor(float(validation_fraction) * n_obs))
    training_size = n_obs - validation_size
    minimum_training_rows = max(20, int(minimum_validation_rows))

    if (
        validation_size < int(minimum_validation_rows)
        or training_size < minimum_training_rows
    ):
        return None

    rng = np.random.default_rng(int(seed))
    order = rng.permutation(n_obs)
    validation_indices = order[:validation_size]
    training_indices = order[validation_size:]
    return (
        data.iloc[training_indices].reset_index(drop=True),
        data.iloc[validation_indices].reset_index(drop=True),
    )


@dataclass
class FittedExplicitDvine4:
    """One fitted order ``P1 -- P2 -- P3 -- child``."""

    child: str
    first_parent: str
    second_parent: str
    third_parent: str

    c_first_second: Any
    c_second_third: Any
    c_third_child: Any
    c_first_third_given_second: Any
    c_second_child_given_third: Any
    c_first_child_given_second_third: Any

    data_first_second: np.ndarray
    data_second_third: np.ndarray
    data_third_child: np.ndarray
    data_first_third_given_second: np.ndarray
    data_second_child_given_third: np.ndarray
    data_first_child_given_second_third: np.ndarray

    n_obs: int
    npars: float
    loglik: float
    aic: float
    bic: float
    conditional_npars: float
    conditional_loglik: float
    conditional_aic: float
    conditional_bic: float

    @property
    def parent_order(self) -> list[str]:
        return [self.first_parent, self.second_parent, self.third_parent]

    @property
    def order(self) -> list[str]:
        return [*self.parent_order, self.child]

    @property
    def order_key(self) -> str:
        return _order_key(self.parent_order, self.child)


def fit_explicit_dvine4_order(
    u_df: pd.DataFrame,
    child: str,
    first_parent: str,
    second_parent: str,
    third_parent: str,
    controls: Any,
    clip_eps: float = 1e-6,
) -> FittedExplicitDvine4:
    """Fit the explicit D-vine ``first--second--third--child``."""
    variables = [first_parent, second_parent, third_parent, child]
    if len(set(variables)) != 4:
        raise ValueError("The three parents and child must be distinct variables.")
    missing = set(variables).difference(u_df.columns)
    if missing:
        raise KeyError(
            "Missing variables required for four-dimensional D-vine fitting: "
            + ", ".join(sorted(missing))
        )
    if len(u_df) < 5:
        raise ValueError("At least five observations are required for fitting.")

    u1 = _clip_u(u_df[first_parent].to_numpy(dtype=float), eps=clip_eps)
    u2 = _clip_u(u_df[second_parent].to_numpy(dtype=float), eps=clip_eps)
    u3 = _clip_u(u_df[third_parent].to_numpy(dtype=float), eps=clip_eps)
    ux = _clip_u(u_df[child].to_numpy(dtype=float), eps=clip_eps)

    data12 = np.asfortranarray(np.column_stack([u1, u2]), dtype=float)
    data23 = np.asfortranarray(np.column_stack([u2, u3]), dtype=float)
    data3x = np.asfortranarray(np.column_stack([u3, ux]), dtype=float)

    c12 = _fit_bicop(data12, controls=controls)
    c23 = _fit_bicop(data23, controls=controls)
    c3x = _fit_bicop(data3x, controls=controls)

    # Tree 2, parent side: C(first, third | second).
    w1_given_2 = _clip_u(c12.hfunc2(data12), eps=clip_eps)
    w3_given_2 = _clip_u(c23.hfunc1(data23), eps=clip_eps)
    data13_given_2 = np.asfortranarray(
        np.column_stack([w1_given_2, w3_given_2]), dtype=float
    )
    c13_given_2 = _fit_bicop(data13_given_2, controls=controls)

    # Tree 2, child side: C(second, child | third).
    w2_given_3 = _clip_u(c23.hfunc2(data23), eps=clip_eps)
    wx_given_3 = _clip_u(c3x.hfunc1(data3x), eps=clip_eps)
    data2x_given_3 = np.asfortranarray(
        np.column_stack([w2_given_3, wx_given_3]), dtype=float
    )
    c2x_given_3 = _fit_bicop(data2x_given_3, controls=controls)

    # Tree 3: C(first, child | second, third).
    w1_given_23 = _clip_u(c13_given_2.hfunc2(data13_given_2), eps=clip_eps)
    wx_given_23 = _clip_u(c2x_given_3.hfunc1(data2x_given_3), eps=clip_eps)
    data1x_given_23 = np.asfortranarray(
        np.column_stack([w1_given_23, wx_given_23]), dtype=float
    )
    c1x_given_23 = _fit_bicop(data1x_given_23, controls=controls)

    all_models_and_data = [
        (c12, data12),
        (c23, data23),
        (c3x, data3x),
        (c13_given_2, data13_given_2),
        (c2x_given_3, data2x_given_3),
        (c1x_given_23, data1x_given_23),
    ]
    child_models_and_data = [
        (c3x, data3x),
        (c2x_given_3, data2x_given_3),
        (c1x_given_23, data1x_given_23),
    ]

    loglik = float(sum(_model_loglik(m, d) for m, d in all_models_and_data))
    npars = float(sum(_model_npars(m) for m, _ in all_models_and_data))
    conditional_loglik = float(
        sum(_model_loglik(m, d) for m, d in child_models_and_data)
    )
    conditional_npars = float(
        sum(_model_npars(m) for m, _ in child_models_and_data)
    )
    n_obs = len(u_df)

    return FittedExplicitDvine4(
        child=child,
        first_parent=first_parent,
        second_parent=second_parent,
        third_parent=third_parent,
        c_first_second=c12,
        c_second_third=c23,
        c_third_child=c3x,
        c_first_third_given_second=c13_given_2,
        c_second_child_given_third=c2x_given_3,
        c_first_child_given_second_third=c1x_given_23,
        data_first_second=data12,
        data_second_third=data23,
        data_third_child=data3x,
        data_first_third_given_second=data13_given_2,
        data_second_child_given_third=data2x_given_3,
        data_first_child_given_second_third=data1x_given_23,
        n_obs=n_obs,
        npars=npars,
        loglik=loglik,
        aic=float(-2.0 * loglik + 2.0 * npars),
        bic=float(-2.0 * loglik + np.log(n_obs) * npars),
        conditional_npars=conditional_npars,
        conditional_loglik=conditional_loglik,
        conditional_aic=float(
            -2.0 * conditional_loglik + 2.0 * conditional_npars
        ),
        conditional_bic=float(
            -2.0 * conditional_loglik
            + np.log(n_obs) * conditional_npars
        ),
    )


def _conditional_coordinates(
    fitted: FittedExplicitDvine4,
    data_df: pd.DataFrame,
    clip_eps: float,
) -> dict[str, np.ndarray]:
    u1 = _clip_u(data_df[fitted.first_parent].to_numpy(dtype=float), clip_eps)
    u2 = _clip_u(data_df[fitted.second_parent].to_numpy(dtype=float), clip_eps)
    u3 = _clip_u(data_df[fitted.third_parent].to_numpy(dtype=float), clip_eps)
    ux = _clip_u(data_df[fitted.child].to_numpy(dtype=float), clip_eps)

    data12 = np.asfortranarray(np.column_stack([u1, u2]), dtype=float)
    data23 = np.asfortranarray(np.column_stack([u2, u3]), dtype=float)
    data3x = np.asfortranarray(np.column_stack([u3, ux]), dtype=float)

    w1_2 = _clip_u(fitted.c_first_second.hfunc2(data12), clip_eps)
    w3_2 = _clip_u(fitted.c_second_third.hfunc1(data23), clip_eps)
    data13_2 = np.asfortranarray(np.column_stack([w1_2, w3_2]), dtype=float)

    w2_3 = _clip_u(fitted.c_second_third.hfunc2(data23), clip_eps)
    wx_3 = _clip_u(fitted.c_third_child.hfunc1(data3x), clip_eps)
    data2x_3 = np.asfortranarray(np.column_stack([w2_3, wx_3]), dtype=float)

    w1_23 = _clip_u(
        fitted.c_first_third_given_second.hfunc2(data13_2), clip_eps
    )
    wx_23 = _clip_u(
        fitted.c_second_child_given_third.hfunc1(data2x_3), clip_eps
    )
    data1x_23 = np.asfortranarray(np.column_stack([w1_23, wx_23]), dtype=float)

    return {
        "data12": data12,
        "data23": data23,
        "data3x": data3x,
        "data13_2": data13_2,
        "data2x_3": data2x_3,
        "data1x_23": data1x_23,
        "w1_23": w1_23,
        "w2_3": w2_3,
        "u3": u3,
    }


def conditional_loglik_values_dvine4(
    fitted: FittedExplicitDvine4,
    data_df: pd.DataFrame,
    clip_eps: float = 1e-6,
) -> np.ndarray:
    """Observation-wise log copula density of child given all three parents."""
    if data_df.empty:
        raise ValueError("Scoring data must not be empty.")
    coordinates = _conditional_coordinates(fitted, data_df, clip_eps)
    values = (
        _density_log_values(fitted.c_third_child, coordinates["data3x"])
        + _density_log_values(
            fitted.c_second_child_given_third,
            coordinates["data2x_3"],
        )
        + _density_log_values(
            fitted.c_first_child_given_second_third,
            coordinates["data1x_23"],
        )
    )
    if not np.isfinite(values).all():
        raise RuntimeError("Conditional log-likelihood contains non-finite values.")
    return values


def parent_loglik_values_dvine4(
    fitted: FittedExplicitDvine4,
    data_df: pd.DataFrame,
    clip_eps: float = 1e-6,
) -> np.ndarray:
    """Observation-wise log density of the three-parent marginal copula."""
    coordinates = _conditional_coordinates(fitted, data_df, clip_eps)
    return (
        _density_log_values(fitted.c_first_second, coordinates["data12"])
        + _density_log_values(fitted.c_second_third, coordinates["data23"])
        + _density_log_values(
            fitted.c_first_third_given_second,
            coordinates["data13_2"],
        )
    )


def joint_loglik_values_dvine4(
    fitted: FittedExplicitDvine4,
    data_df: pd.DataFrame,
    clip_eps: float = 1e-6,
) -> np.ndarray:
    """Observation-wise log density of the complete four-dimensional vine."""
    return parent_loglik_values_dvine4(
        fitted, data_df, clip_eps
    ) + conditional_loglik_values_dvine4(fitted, data_df, clip_eps)


def sample_child_from_fitted_dvine4(
    fitted: FittedExplicitDvine4,
    parent_df: pd.DataFrame,
    seed: int,
    clip_eps: float = 1e-6,
) -> np.ndarray:
    """Sample the child conditionally from an in-memory fitted D-vine."""
    required = fitted.parent_order
    missing = set(required).difference(parent_df.columns)
    if missing:
        raise KeyError("Missing parent columns: " + ", ".join(sorted(missing)))

    u1 = _clip_u(parent_df[fitted.first_parent].to_numpy(dtype=float), clip_eps)
    u2 = _clip_u(parent_df[fitted.second_parent].to_numpy(dtype=float), clip_eps)
    u3 = _clip_u(parent_df[fitted.third_parent].to_numpy(dtype=float), clip_eps)

    data12 = np.asfortranarray(np.column_stack([u1, u2]), dtype=float)
    data23 = np.asfortranarray(np.column_stack([u2, u3]), dtype=float)
    w1_2 = _clip_u(fitted.c_first_second.hfunc2(data12), clip_eps)
    w3_2 = _clip_u(fitted.c_second_third.hfunc1(data23), clip_eps)
    data13_2 = np.asfortranarray(np.column_stack([w1_2, w3_2]), dtype=float)
    w1_23 = _clip_u(
        fitted.c_first_third_given_second.hfunc2(data13_2), clip_eps
    )
    w2_3 = _clip_u(fitted.c_second_third.hfunc2(data23), clip_eps)

    rng = np.random.default_rng(int(seed))
    q = rng.uniform(clip_eps, 1.0 - clip_eps, size=len(parent_df))
    wx_23 = fitted.c_first_child_given_second_third.hinv1(
        np.asfortranarray(np.column_stack([w1_23, q]), dtype=float)
    )
    wx_23 = _clip_u(wx_23, clip_eps)
    wx_3 = fitted.c_second_child_given_third.hinv1(
        np.asfortranarray(np.column_stack([w2_3, wx_23]), dtype=float)
    )
    wx_3 = _clip_u(wx_3, clip_eps)
    ux = fitted.c_third_child.hinv1(
        np.asfortranarray(np.column_stack([u3, wx_3]), dtype=float)
    )
    ux = _clip_u(ux, clip_eps)
    if not np.isfinite(ux).all():
        raise RuntimeError("Conditional samples contain non-finite values.")
    return np.asarray(ux, dtype=float)


@dataclass
class ConditionalVineIndegree3Record:
    dataset_id: str
    child: str
    parent_1: str
    parent_2: str
    parent_3: str
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
    second_best_order_score: float | None
    order_score_margin: float | None

    conditional_model_type: str
    simplifying_assumption_selected: bool

    bicop_parent_1_parent_2_file: str
    bicop_parent_2_parent_3_file: str
    bicop_parent_3_child_file: str
    bicop_parent_1_parent_3_given_parent_2_file: str
    bicop_parent_2_child_given_parent_3_file: str
    bicop_parent_1_child_given_parent_2_parent_3_file: str

    family_parent_1_parent_2: str | None
    family_parent_2_parent_3: str | None
    family_parent_3_child: str | None
    family_parent_1_parent_3_given_parent_2: str | None
    family_parent_2_child_given_parent_3: str | None
    family_parent_1_child_given_parent_2_parent_3: str | None

    rotation_parent_1_parent_2: int | None
    rotation_parent_2_parent_3: int | None
    rotation_parent_3_child: int | None
    rotation_parent_1_parent_3_given_parent_2: int | None
    rotation_parent_2_child_given_parent_3: int | None
    rotation_parent_1_child_given_parent_2_parent_3: int | None

    parameters_parent_1_parent_2: list
    parameters_parent_2_parent_3: list
    parameters_parent_3_child: list
    parameters_parent_1_parent_3_given_parent_2: list
    parameters_parent_2_child_given_parent_3: list
    parameters_parent_1_child_given_parent_2_parent_3: list

    npars: float | None
    loglik: float | None
    aic: float | None
    bic: float | None
    conditional_npars: float | None
    conditional_loglik: float | None
    conditional_aic: float | None
    conditional_bic: float | None

    pair_family_summary: str | None
    vine_structure: str | None
    status: str
    error: str | None = None


@dataclass
class ConditionalVineIndegree3Library:
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
    records: list[ConditionalVineIndegree3Record]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(record) for record in self.records])

    def save_pickle(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset_id": self.dataset_id,
            "columns": self.columns,
            "n_rows": self.n_rows,
            "parent_set_size": 3,
            "selection_criterion": self.selection_criterion,
            "family_set": self.family_set,
            "order_strategy": self.order_strategy,
            "order_validation_fraction": self.order_validation_fraction,
            "order_selection_seed": self.order_selection_seed,
            "minimum_validation_rows": self.minimum_validation_rows,
            "conditional_model_strategy": "simplified",
            "records": [asdict(record) for record in self.records],
            "implementation": "explicit_4d_dvine_six_parent_orders_v1",
        }
        with path.open("wb") as file:
            pickle.dump(payload, file)

    @staticmethod
    def load_pickle(path: str | Path) -> dict[str, Any]:
        with Path(path).open("rb") as file:
            return pickle.load(file)


def _select_order(
    u_df: pd.DataFrame,
    child: str,
    parents: Sequence[str],
    controls: Any,
    order_strategy: OrderStrategy,
    validation_fraction: float,
    selection_seed: int,
    minimum_validation_rows: int,
    clip_eps: float,
) -> tuple[
    tuple[str, str, str],
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
    canonical = _canonical_parent_set(parents)
    candidate_orders = list(permutations(canonical))
    keys = [_order_key(order, child) for order in candidate_orders]
    scores: dict[str, float | None] = {key: None for key in keys}
    statuses: dict[str, str] = {key: "not_evaluated" for key in keys}
    errors: dict[str, str | None] = {key: None for key in keys}

    if order_strategy == "fixed":
        selected = tuple(map(str, parents))
        if len(selected) != 3 or len(set(selected)) != 3:
            raise ValueError("Fixed order requires three distinct parents.")
        key = _order_key(selected, child)
        statuses[key] = "selected_without_comparison"
        return (
            selected, "fixed_parent_order", "not_applicable", scores,
            statuses, errors, None, None, None, None, None,
        )

    split = None
    if order_strategy == "heldout_conditional_loglik":
        split = _make_train_validation_split(
            u_df,
            validation_fraction=validation_fraction,
            seed=selection_seed,
            minimum_validation_rows=minimum_validation_rows,
        )

    if split is not None:
        fit_df, score_df = split
        method = "heldout_conditional_loglik"
        metric = "mean_conditional_loglik"
        training_size = len(fit_df)
        validation_size = len(score_df)

        def score_function(model: FittedExplicitDvine4) -> float:
            return float(
                np.mean(
                    conditional_loglik_values_dvine4(
                        model, score_df, clip_eps=clip_eps
                    )
                )
            )

    else:
        fit_df = u_df
        method = (
            "bic" if order_strategy == "bic"
            else "bic_fallback_insufficient_validation_rows"
        )
        metric = "negative_conditional_bic"
        training_size = len(u_df)
        validation_size = None

        def score_function(model: FittedExplicitDvine4) -> float:
            return -float(model.conditional_bic)

    successful: list[tuple[tuple[str, str, str], float]] = []
    for order in candidate_orders:
        key = _order_key(order, child)
        try:
            fitted = fit_explicit_dvine4_order(
                u_df=fit_df,
                child=child,
                first_parent=order[0],
                second_parent=order[1],
                third_parent=order[2],
                controls=controls,
                clip_eps=clip_eps,
            )
            score = float(score_function(fitted))
            if not np.isfinite(score):
                raise RuntimeError("Candidate order score is not finite.")
            scores[key] = score
            statuses[key] = "ok"
            successful.append((order, score))
        except Exception as exc:
            statuses[key] = "failed"
            errors[key] = f"{type(exc).__name__}: {exc}"

    if not successful:
        raise RuntimeError(f"All six candidate parent orders failed: {errors}")

    successful.sort(key=lambda item: (-item[1], *item[0]))
    selected, selected_score = successful[0]
    second_score = successful[1][1] if len(successful) > 1 else None
    margin = (
        float(selected_score - second_score)
        if second_score is not None else None
    )
    return (
        selected,
        method,
        metric,
        scores,
        statuses,
        errors,
        float(selected_score),
        float(second_score) if second_score is not None else None,
        margin,
        training_size,
        validation_size,
    )


def fit_one_conditional_vine_indegree3(
    u_df: pd.DataFrame,
    child: str,
    parents: Sequence[str],
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
) -> ConditionalVineIndegree3Record:
    """Select among six parent orders and fit the final four-dimensional vine."""
    strategy = _validate_order_strategy(order_strategy)
    canonical = _canonical_parent_set(parents)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    selected = canonical
    method = "not_completed"
    metric = "not_completed"
    scores: dict[str, float | None] = {}
    statuses: dict[str, str] = {}
    errors: dict[str, str | None] = {}
    selected_score = None
    second_score = None
    margin = None
    training_size = None
    validation_size = None

    try:
        (
            selected,
            method,
            metric,
            scores,
            statuses,
            errors,
            selected_score,
            second_score,
            margin,
            training_size,
            validation_size,
        ) = _select_order(
            u_df=u_df,
            child=child,
            parents=parents,
            controls=controls,
            order_strategy=strategy,
            validation_fraction=order_validation_fraction,
            selection_seed=order_selection_seed,
            minimum_validation_rows=minimum_validation_rows,
            clip_eps=clip_eps,
        )

        final = fit_explicit_dvine4_order(
            u_df=u_df,
            child=child,
            first_parent=selected[0],
            second_parent=selected[1],
            third_parent=selected[2],
            controls=controls,
            clip_eps=clip_eps,
        )

        model_name = (
            f"child_{_safe_filename(child)}__given__"
            f"{_safe_filename(selected[0])}__{_safe_filename(selected[1])}__"
            f"{_safe_filename(selected[2])}"
        )
        paths = {
            "c12": model_dir / f"{model_name}__c_parent1_parent2.json",
            "c23": model_dir / f"{model_name}__c_parent2_parent3.json",
            "c3x": model_dir / f"{model_name}__c_parent3_child.json",
            "c13_2": model_dir / (
                f"{model_name}__c_parent1_parent3_given_parent2.json"
            ),
            "c2x_3": model_dir / (
                f"{model_name}__c_parent2_child_given_parent3.json"
            ),
            "c1x_23": model_dir / (
                f"{model_name}__c_parent1_child_given_parent2_parent3.json"
            ),
        }
        files = {
            "c12": _save_bicop(final.c_first_second, paths["c12"]),
            "c23": _save_bicop(final.c_second_third, paths["c23"]),
            "c3x": _save_bicop(final.c_third_child, paths["c3x"]),
            "c13_2": _save_bicop(
                final.c_first_third_given_second, paths["c13_2"]
            ),
            "c2x_3": _save_bicop(
                final.c_second_child_given_third, paths["c2x_3"]
            ),
            "c1x_23": _save_bicop(
                final.c_first_child_given_second_third, paths["c1x_23"]
            ),
        }
        summaries = {
            "c12": _bicop_summary(final.c_first_second),
            "c23": _bicop_summary(final.c_second_third),
            "c3x": _bicop_summary(final.c_third_child),
            "c13_2": _bicop_summary(final.c_first_third_given_second),
            "c2x_3": _bicop_summary(final.c_second_child_given_third),
            "c1x_23": _bicop_summary(
                final.c_first_child_given_second_third
            ),
        }

        return ConditionalVineIndegree3Record(
            dataset_id=dataset_id,
            child=child,
            parent_1=selected[0],
            parent_2=selected[1],
            parent_3=selected[2],
            parent_set=list(canonical),
            selected_order=[*selected, child],
            variables=[*selected, child],
            parent_set_size=3,
            n_obs=len(u_df),
            n_variables=4,
            selection_criterion=selection_criterion,
            candidate_families=candidate_families,
            order_strategy_requested=strategy,
            order_selection_method=method,
            order_selection_metric=metric,
            order_validation_fraction=(
                float(order_validation_fraction)
                if method.startswith("heldout") else None
            ),
            order_selection_seed=(
                int(order_selection_seed) if method.startswith("heldout") else None
            ),
            order_training_size=training_size,
            order_validation_size=validation_size,
            minimum_validation_rows=int(minimum_validation_rows),
            candidate_order_scores=scores,
            candidate_order_statuses=statuses,
            candidate_order_errors=errors,
            selected_order_score=selected_score,
            second_best_order_score=second_score,
            order_score_margin=margin,
            conditional_model_type="simplified_4d_dvine",
            simplifying_assumption_selected=True,
            bicop_parent_1_parent_2_file=files["c12"],
            bicop_parent_2_parent_3_file=files["c23"],
            bicop_parent_3_child_file=files["c3x"],
            bicop_parent_1_parent_3_given_parent_2_file=files["c13_2"],
            bicop_parent_2_child_given_parent_3_file=files["c2x_3"],
            bicop_parent_1_child_given_parent_2_parent_3_file=files["c1x_23"],
            family_parent_1_parent_2=summaries["c12"]["family"],
            family_parent_2_parent_3=summaries["c23"]["family"],
            family_parent_3_child=summaries["c3x"]["family"],
            family_parent_1_parent_3_given_parent_2=summaries["c13_2"]["family"],
            family_parent_2_child_given_parent_3=summaries["c2x_3"]["family"],
            family_parent_1_child_given_parent_2_parent_3=summaries["c1x_23"]["family"],
            rotation_parent_1_parent_2=summaries["c12"]["rotation"],
            rotation_parent_2_parent_3=summaries["c23"]["rotation"],
            rotation_parent_3_child=summaries["c3x"]["rotation"],
            rotation_parent_1_parent_3_given_parent_2=summaries["c13_2"]["rotation"],
            rotation_parent_2_child_given_parent_3=summaries["c2x_3"]["rotation"],
            rotation_parent_1_child_given_parent_2_parent_3=summaries["c1x_23"]["rotation"],
            parameters_parent_1_parent_2=summaries["c12"]["parameters"],
            parameters_parent_2_parent_3=summaries["c23"]["parameters"],
            parameters_parent_3_child=summaries["c3x"]["parameters"],
            parameters_parent_1_parent_3_given_parent_2=summaries["c13_2"]["parameters"],
            parameters_parent_2_child_given_parent_3=summaries["c2x_3"]["parameters"],
            parameters_parent_1_child_given_parent_2_parent_3=summaries["c1x_23"]["parameters"],
            npars=final.npars,
            loglik=final.loglik,
            aic=final.aic,
            bic=final.bic,
            conditional_npars=final.conditional_npars,
            conditional_loglik=final.conditional_loglik,
            conditional_aic=final.conditional_aic,
            conditional_bic=final.conditional_bic,
            pair_family_summary=(
                f"C({selected[0]},{selected[1]})={summaries['c12']['family']}; "
                f"C({selected[1]},{selected[2]})={summaries['c23']['family']}; "
                f"C({selected[2]},{child})={summaries['c3x']['family']}; "
                f"C({selected[0]},{selected[2]}|{selected[1]})="
                f"{summaries['c13_2']['family']}; "
                f"C({selected[1]},{child}|{selected[2]})="
                f"{summaries['c2x_3']['family']}; "
                f"C({selected[0]},{child}|{selected[1]},{selected[2]})="
                f"{summaries['c1x_23']['family']}"
            ),
            vine_structure=(
                f"explicit D-vine: {selected[0]} -- {selected[1]} -- "
                f"{selected[2]} -- {child}"
            ),
            status="ok",
            error=None,
        )
    except Exception as exc:
        selected_list = list(selected)
        return ConditionalVineIndegree3Record(
            dataset_id=dataset_id,
            child=child,
            parent_1=selected_list[0],
            parent_2=selected_list[1],
            parent_3=selected_list[2],
            parent_set=list(canonical),
            selected_order=[*selected_list, child],
            variables=[*selected_list, child],
            parent_set_size=3,
            n_obs=len(u_df),
            n_variables=4,
            selection_criterion=selection_criterion,
            candidate_families=candidate_families,
            order_strategy_requested=strategy,
            order_selection_method=method,
            order_selection_metric=metric,
            order_validation_fraction=(
                float(order_validation_fraction)
                if method.startswith("heldout") else None
            ),
            order_selection_seed=(
                int(order_selection_seed) if method.startswith("heldout") else None
            ),
            order_training_size=training_size,
            order_validation_size=validation_size,
            minimum_validation_rows=int(minimum_validation_rows),
            candidate_order_scores=scores,
            candidate_order_statuses=statuses,
            candidate_order_errors=errors,
            selected_order_score=selected_score,
            second_best_order_score=second_score,
            order_score_margin=margin,
            conditional_model_type="failed",
            simplifying_assumption_selected=True,
            bicop_parent_1_parent_2_file="",
            bicop_parent_2_parent_3_file="",
            bicop_parent_3_child_file="",
            bicop_parent_1_parent_3_given_parent_2_file="",
            bicop_parent_2_child_given_parent_3_file="",
            bicop_parent_1_child_given_parent_2_parent_3_file="",
            family_parent_1_parent_2=None,
            family_parent_2_parent_3=None,
            family_parent_3_child=None,
            family_parent_1_parent_3_given_parent_2=None,
            family_parent_2_child_given_parent_3=None,
            family_parent_1_child_given_parent_2_parent_3=None,
            rotation_parent_1_parent_2=None,
            rotation_parent_2_parent_3=None,
            rotation_parent_3_child=None,
            rotation_parent_1_parent_3_given_parent_2=None,
            rotation_parent_2_child_given_parent_3=None,
            rotation_parent_1_child_given_parent_2_parent_3=None,
            parameters_parent_1_parent_2=[],
            parameters_parent_2_parent_3=[],
            parameters_parent_3_child=[],
            parameters_parent_1_parent_3_given_parent_2=[],
            parameters_parent_2_child_given_parent_3=[],
            parameters_parent_1_child_given_parent_2_parent_3=[],
            npars=None,
            loglik=None,
            aic=None,
            bic=None,
            conditional_npars=None,
            conditional_loglik=None,
            conditional_aic=None,
            conditional_bic=None,
            pair_family_summary=None,
            vine_structure=None,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )


def build_conditional_vine_library_indegree3(
    u_df: pd.DataFrame,
    dataset_id: str,
    selection_criterion: str = "bic",
    allow_rotations: bool = True,
    num_threads: int = 1,
    model_dir: str | Path = "artifacts/conditional_vines_indegree3",
    order_strategy: str = "heldout_conditional_loglik",
    order_validation_fraction: float = 0.20,
    order_selection_seed: int = 42,
    minimum_validation_rows: int = 50,
    limit: int | None = None,
) -> ConditionalVineIndegree3Library:
    """Fit every child/three-parent-set mechanism in a data table."""
    if u_df.empty:
        raise ValueError("Cannot build an indegree-three library from empty data.")
    strategy = _validate_order_strategy(order_strategy)
    controls, family_names = build_bicop_fit_controls(
        selection_criterion=selection_criterion,
        allow_rotations=allow_rotations,
        num_threads=num_threads,
    )
    columns = list(u_df.columns)
    tasks: list[tuple[str, tuple[str, str, str]]] = []
    for child in columns:
        possible = [column for column in columns if column != child]
        for parent_set in combinations(possible, 3):
            tasks.append((child, parent_set))
    if limit is not None:
        if int(limit) < 1:
            raise ValueError("limit must be positive when provided.")
        tasks = tasks[: int(limit)]

    records: list[ConditionalVineIndegree3Record] = []
    total = len(tasks)
    for index, (child, parents) in enumerate(tasks, start=1):
        print(
            f"[{index:04d}/{total:04d}] Fitting four-dimensional D-vine: "
            f"{child} | {', '.join(parents)}"
        )
        records.append(
            fit_one_conditional_vine_indegree3(
                u_df=u_df,
                child=child,
                parents=parents,
                dataset_id=dataset_id,
                controls=controls,
                candidate_families=family_names,
                selection_criterion=selection_criterion,
                model_dir=model_dir,
                order_strategy=strategy,
                order_validation_fraction=order_validation_fraction,
                order_selection_seed=order_selection_seed,
                minimum_validation_rows=minimum_validation_rows,
            )
        )

    return ConditionalVineIndegree3Library(
        dataset_id=dataset_id,
        columns=columns,
        n_rows=len(u_df),
        parent_set_size=3,
        selection_criterion=selection_criterion,
        family_set=family_names,
        order_strategy=strategy,
        order_validation_fraction=float(order_validation_fraction),
        order_selection_seed=int(order_selection_seed),
        minimum_validation_rows=int(minimum_validation_rows),
        records=records,
    )
