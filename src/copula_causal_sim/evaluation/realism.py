from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Literal

import numpy as np
import pandas as pd


DEFAULT_QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]
DEFAULT_TAIL_PROBABILITY = 0.05


TAIL_DEPENDENCE_COLUMNS = [
    "var1",
    "var2",
    "tail_probability",
    "lower_threshold",
    "upper_threshold",
    "is_graph_edge_pair",
    "real_n_obs",
    "generated_n_obs",
    "real_lower_joint_count",
    "generated_lower_joint_count",
    "real_upper_joint_count",
    "generated_upper_joint_count",
    "real_lower_joint_probability",
    "generated_lower_joint_probability",
    "real_upper_joint_probability",
    "generated_upper_joint_probability",
    "real_lower_tail_dependence",
    "generated_lower_tail_dependence",
    "lower_tail_absolute_error",
    "real_upper_tail_dependence",
    "generated_upper_tail_dependence",
    "upper_tail_absolute_error",
]


def validate_same_columns(real_df: pd.DataFrame, generated_df: pd.DataFrame) -> None:
    real_cols = list(real_df.columns)
    gen_cols = list(generated_df.columns)

    if real_cols != gen_cols:
        raise ValueError(
            "Column mismatch between real and generated data.\n"
            f"Real columns: {real_cols}\n"
            f"Generated columns: {gen_cols}"
        )


def _variable_scale(values: pd.Series, eps: float = 1e-12) -> tuple[float, str]:
    """
    Choose a scale for normalized marginal errors.

    Priority:
    1. IQR
    2. standard deviation
    3. 1.0 fallback
    """
    q25 = float(values.quantile(0.25))
    q75 = float(values.quantile(0.75))
    iqr = q75 - q25

    if abs(iqr) > eps:
        return float(iqr), "iqr"

    std = float(values.std())

    if abs(std) > eps:
        return float(std), "std"

    return 1.0, "constant_fallback"


def _as_float(value: Any) -> float:
    """Convert a scalar value from pandas or numpy to a Python float."""
    if isinstance(value, np.generic):
        return float(value.item())
    return float(value)


def compute_marginal_quantile_errors(
    real_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    quantiles: list[float] | None = None,
) -> pd.DataFrame:
    """
    Compare marginal empirical quantiles between real and generated data.

    Returns both raw absolute error and normalized absolute error.

    Normalized error:
        abs(real_quantile - generated_quantile) / scale(real_variable)

    where scale is IQR by default, with standard-deviation fallback.
    """
    validate_same_columns(real_df, generated_df)

    if quantiles is None:
        quantiles = DEFAULT_QUANTILES

    rows = []

    for col in real_df.columns:
        scale, scale_type = _variable_scale(real_df[col])

        for q in quantiles:
            real_value = float(real_df[col].quantile(q))
            generated_value = float(generated_df[col].quantile(q))
            absolute_error = abs(real_value - generated_value)

            rows.append({
                "variable": col,
                "quantile": q,
                "real_value": real_value,
                "generated_value": generated_value,
                "absolute_error": absolute_error,
                "normalization_scale": scale,
                "normalization_scale_type": scale_type,
                "normalized_absolute_error": absolute_error / scale,
            })

    return pd.DataFrame(rows)


def summarize_marginal_errors(marginal_errors: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize marginal quantile errors per variable.
    """
    summary = (
        marginal_errors
        .groupby("variable", as_index=False)
        .agg(
            mean_abs_quantile_error=("absolute_error", "mean"),
            median_abs_quantile_error=("absolute_error", "median"),
            max_abs_quantile_error=("absolute_error", "max"),
            mean_normalized_abs_quantile_error=("normalized_absolute_error", "mean"),
            median_normalized_abs_quantile_error=("normalized_absolute_error", "median"),
            max_normalized_abs_quantile_error=("normalized_absolute_error", "max"),
            normalization_scale=("normalization_scale", "first"),
            normalization_scale_type=("normalization_scale_type", "first"),
        )
        .sort_values("mean_normalized_abs_quantile_error", ascending=False)
    )

    return summary


def correlation_matrix(
    df: pd.DataFrame,
    method: Literal["kendall", "spearman", "pearson"],
) -> pd.DataFrame:
    """
    Compute dependence matrix using pandas correlation methods.

    Supported methods:
        - kendall
        - spearman
        - pearson
    """
    if method not in {"kendall", "spearman", "pearson"}:
        raise ValueError(f"Unsupported correlation method: {method}")

    return df.corr(method=method)


def matrix_absolute_error(
    real_matrix: pd.DataFrame,
    generated_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Absolute elementwise error between two square dependence matrices.
    """
    if list(real_matrix.columns) != list(generated_matrix.columns):
        raise ValueError("Matrix column mismatch.")

    if list(real_matrix.index) != list(generated_matrix.index):
        raise ValueError("Matrix index mismatch.")

    return (real_matrix - generated_matrix).abs()


def summarize_matrix_error(error_matrix: pd.DataFrame) -> dict[str, float]:
    """
    Summarize off-diagonal entries of an absolute error matrix.
    """
    values = error_matrix.to_numpy(dtype=float)

    mask = ~np.eye(values.shape[0], dtype=bool)
    off_diag = values[mask]

    return {
        "mean_abs_error": float(np.nanmean(off_diag)),
        "median_abs_error": float(np.nanmedian(off_diag)),
        "max_abs_error": float(np.nanmax(off_diag)),
    }


def top_pairwise_errors(
    error_matrix: pd.DataFrame,
    real_matrix: pd.DataFrame,
    generated_matrix: pd.DataFrame,
    top_k: int = 20,
) -> pd.DataFrame:
    """
    Return largest off-diagonal dependence errors.
    """
    rows = []
    columns = list(error_matrix.columns)

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            var1 = columns[i]
            var2 = columns[j]

            rows.append({
                "var1": var1,
                "var2": var2,
                "real_value": _as_float(real_matrix.loc[var1, var2]),
                "generated_value": _as_float(generated_matrix.loc[var1, var2]),
                "absolute_error": _as_float(error_matrix.loc[var1, var2]),
            })

    return (
        pd.DataFrame(rows)
        .sort_values("absolute_error", ascending=False)
        .head(top_k)
    )


def _safe_mean(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None

    return float(np.mean(finite))


def _safe_median(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None

    return float(np.median(finite))


def _safe_max(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None

    return float(np.max(finite))


def graph_pairwise_dependence_table(
    real_matrix: pd.DataFrame,
    generated_matrix: pd.DataFrame,
    adjacency: np.ndarray,
) -> pd.DataFrame:
    """
    Build a pairwise graph-aware dependence table.

    Edge relation is treated as undirected for pairwise dependence summaries:
        edge_pair = 1 if i -> j or j -> i.
    """
    columns = list(real_matrix.columns)

    if list(generated_matrix.columns) != columns:
        raise ValueError("Generated matrix columns do not match real matrix columns.")

    adj = np.asarray(adjacency, dtype=int)

    if adj.shape != (len(columns), len(columns)):
        raise ValueError(
            "Adjacency shape does not match dependence matrix size.\n"
            f"Adjacency shape: {adj.shape}\n"
            f"Expected: {(len(columns), len(columns))}"
        )

    undirected_edge = ((adj + adj.T) > 0)

    rows = []

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            var1 = columns[i]
            var2 = columns[j]

            is_edge = bool(undirected_edge[i, j])

            real_value = _as_float(real_matrix.loc[var1, var2])
            generated_value = _as_float(generated_matrix.loc[var1, var2])

            rows.append({
                "var1": var1,
                "var2": var2,
                "is_graph_edge_pair": int(is_edge),
                "real_value": real_value,
                "generated_value": generated_value,
                "real_abs_value": abs(real_value),
                "generated_abs_value": abs(generated_value),
                "absolute_error": abs(real_value - generated_value),
            })

    return pd.DataFrame(rows)


def summarize_graph_dependence(pairwise_table: pd.DataFrame) -> dict[str, Any]:
    """
    Summarize edge and non-edge dependence.

    Important interpretation:
    - edge_generated_mean_abs_value tells how much dependence the generator created
      along graph edges.
    - nonedge_generated_mean_abs_value tells how much dependence remains among
      non-edge pairs.
    - generated_edge_nonedge_abs_gap should usually be positive for a graph-conditioned
      generator.
    - edge_mean_abs_error compares generated edge dependence to real-data pairwise
      dependence for the same pairs.
    """
    edge = pairwise_table[pairwise_table["is_graph_edge_pair"] == 1]
    nonedge = pairwise_table[pairwise_table["is_graph_edge_pair"] == 0]

    edge_generated = edge["generated_abs_value"].to_numpy(dtype=float)
    nonedge_generated = nonedge["generated_abs_value"].to_numpy(dtype=float)

    edge_real = edge["real_abs_value"].to_numpy(dtype=float)
    nonedge_real = nonedge["real_abs_value"].to_numpy(dtype=float)

    edge_error = edge["absolute_error"].to_numpy(dtype=float)
    nonedge_error = nonedge["absolute_error"].to_numpy(dtype=float)

    edge_generated_mean = _safe_mean(edge_generated)
    nonedge_generated_mean = _safe_mean(nonedge_generated)

    edge_real_mean = _safe_mean(edge_real)
    nonedge_real_mean = _safe_mean(nonedge_real)

    return {
        "num_edge_pairs": int(len(edge)),
        "num_nonedge_pairs": int(len(nonedge)),

        "edge_real_mean_abs_value": edge_real_mean,
        "edge_generated_mean_abs_value": edge_generated_mean,
        "edge_generated_median_abs_value": _safe_median(edge_generated),
        "edge_generated_max_abs_value": _safe_max(edge_generated),

        "nonedge_real_mean_abs_value": nonedge_real_mean,
        "nonedge_generated_mean_abs_value": nonedge_generated_mean,
        "nonedge_generated_median_abs_value": _safe_median(nonedge_generated),
        "nonedge_generated_max_abs_value": _safe_max(nonedge_generated),

        "edge_mean_abs_error": _safe_mean(edge_error),
        "edge_median_abs_error": _safe_median(edge_error),
        "edge_max_abs_error": _safe_max(edge_error),

        "nonedge_mean_abs_error": _safe_mean(nonedge_error),
        "nonedge_median_abs_error": _safe_median(nonedge_error),
        "nonedge_max_abs_error": _safe_max(nonedge_error),

        "real_edge_nonedge_abs_gap": (
            None
            if edge_real_mean is None or nonedge_real_mean is None
            else edge_real_mean - nonedge_real_mean
        ),
        "generated_edge_nonedge_abs_gap": (
            None
            if edge_generated_mean is None or nonedge_generated_mean is None
            else edge_generated_mean - nonedge_generated_mean
        ),
    }


# -----------------------------------------------------------------------------
# Empirical lower- and upper-tail dependence
# -----------------------------------------------------------------------------


def _validate_tail_probability(tail_probability: float) -> float:
    try:
        value = float(tail_probability)
    except (TypeError, ValueError) as exc:
        raise ValueError("tail_probability must be a number.") from exc

    if not np.isfinite(value):
        raise ValueError("tail_probability must be finite.")

    if not 0.0 < value < 0.5:
        raise ValueError(
            "tail_probability must satisfy 0 < tail_probability < 0.5."
        )

    return value


def _validate_unit_interval_dataframe(
    df: pd.DataFrame,
    dataframe_name: str,
) -> None:
    if df.empty:
        raise ValueError(f"{dataframe_name} is empty.")

    non_numeric = [
        column
        for column in df.columns
        if not pd.api.types.is_numeric_dtype(df[column])
    ]

    if non_numeric:
        raise ValueError(
            f"{dataframe_name} contains non-numeric columns: {non_numeric}"
        )

    values = df.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            f"{dataframe_name} contains NaN or infinite values."
        )

    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError(
            f"{dataframe_name} must contain pseudo-observations in [0, 1]."
        )


def empirical_tail_dependence(
    first: pd.Series | np.ndarray,
    second: pd.Series | np.ndarray,
    tail_probability: float = DEFAULT_TAIL_PROBABILITY,
) -> dict[str, float | int]:
    """
    Estimate lower- and upper-tail dependence for one variable pair.

    For p = tail_probability:

        lower = P(U <= p, V <= p) / p
        upper = P(U >= 1-p, V >= 1-p) / p

    The estimator is intended for copula-scale pseudo-observations.
    """
    p = _validate_tail_probability(tail_probability)

    first_values = np.asarray(first, dtype=float).reshape(-1)
    second_values = np.asarray(second, dtype=float).reshape(-1)

    if first_values.size != second_values.size:
        raise ValueError(
            "Tail-dependence inputs must have the same number of observations."
        )

    if first_values.size == 0:
        raise ValueError("Tail-dependence inputs must not be empty.")

    if not np.isfinite(first_values).all() or not np.isfinite(second_values).all():
        raise ValueError("Tail-dependence inputs contain NaN or infinite values.")

    if (
        ((first_values < 0.0) | (first_values > 1.0)).any()
        or ((second_values < 0.0) | (second_values > 1.0)).any()
    ):
        raise ValueError(
            "Tail-dependence inputs must contain pseudo-observations in [0, 1]."
        )

    upper_threshold = 1.0 - p

    lower_joint = (first_values <= p) & (second_values <= p)
    upper_joint = (
        (first_values >= upper_threshold)
        & (second_values >= upper_threshold)
    )

    n_obs = int(first_values.size)
    lower_joint_count = int(np.count_nonzero(lower_joint))
    upper_joint_count = int(np.count_nonzero(upper_joint))

    lower_joint_probability = lower_joint_count / n_obs
    upper_joint_probability = upper_joint_count / n_obs

    return {
        "n_obs": n_obs,
        "tail_probability": p,
        "lower_threshold": p,
        "upper_threshold": upper_threshold,
        "lower_joint_count": lower_joint_count,
        "upper_joint_count": upper_joint_count,
        "lower_joint_probability": float(lower_joint_probability),
        "upper_joint_probability": float(upper_joint_probability),
        "lower_tail_dependence": float(lower_joint_probability / p),
        "upper_tail_dependence": float(upper_joint_probability / p),
    }


def tail_dependence_table(
    real_u_df: pd.DataFrame,
    generated_u_df: pd.DataFrame,
    adjacency: np.ndarray | None = None,
    tail_probability: float = DEFAULT_TAIL_PROBABILITY,
) -> pd.DataFrame:
    """
    Compare empirical pairwise tail dependence on copula scale.

    The two dataframes may have different row counts, but they must have the
    same ordered columns. If adjacency is provided, each unordered variable
    pair is marked as an edge pair when either i -> j or j -> i is present.
    """
    validate_same_columns(real_u_df, generated_u_df)
    _validate_unit_interval_dataframe(real_u_df, "real_u_df")
    _validate_unit_interval_dataframe(generated_u_df, "generated_u_df")

    p = _validate_tail_probability(tail_probability)
    columns = list(real_u_df.columns)

    undirected_edge: np.ndarray | None = None

    if adjacency is not None:
        adj = np.asarray(adjacency, dtype=int)
        expected_shape = (len(columns), len(columns))

        if adj.shape != expected_shape:
            raise ValueError(
                "Adjacency shape does not match number of variables.\n"
                f"Adjacency shape: {adj.shape}\n"
                f"Expected: {expected_shape}"
            )

        undirected_edge = (adj + adj.T) > 0

    rows: list[dict[str, Any]] = []

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            var1 = columns[i]
            var2 = columns[j]

            real_tail = empirical_tail_dependence(
                real_u_df[var1],
                real_u_df[var2],
                tail_probability=p,
            )

            generated_tail = empirical_tail_dependence(
                generated_u_df[var1],
                generated_u_df[var2],
                tail_probability=p,
            )

            is_edge: int | None
            if undirected_edge is None:
                is_edge = None
            else:
                is_edge = int(bool(undirected_edge[i, j]))

            real_lower = float(real_tail["lower_tail_dependence"])
            generated_lower = float(generated_tail["lower_tail_dependence"])
            real_upper = float(real_tail["upper_tail_dependence"])
            generated_upper = float(generated_tail["upper_tail_dependence"])

            rows.append({
                "var1": var1,
                "var2": var2,
                "tail_probability": p,
                "lower_threshold": p,
                "upper_threshold": 1.0 - p,
                "is_graph_edge_pair": is_edge,
                "real_n_obs": int(real_tail["n_obs"]),
                "generated_n_obs": int(generated_tail["n_obs"]),
                "real_lower_joint_count": int(real_tail["lower_joint_count"]),
                "generated_lower_joint_count": int(
                    generated_tail["lower_joint_count"]
                ),
                "real_upper_joint_count": int(real_tail["upper_joint_count"]),
                "generated_upper_joint_count": int(
                    generated_tail["upper_joint_count"]
                ),
                "real_lower_joint_probability": float(
                    real_tail["lower_joint_probability"]
                ),
                "generated_lower_joint_probability": float(
                    generated_tail["lower_joint_probability"]
                ),
                "real_upper_joint_probability": float(
                    real_tail["upper_joint_probability"]
                ),
                "generated_upper_joint_probability": float(
                    generated_tail["upper_joint_probability"]
                ),
                "real_lower_tail_dependence": real_lower,
                "generated_lower_tail_dependence": generated_lower,
                "lower_tail_absolute_error": abs(real_lower - generated_lower),
                "real_upper_tail_dependence": real_upper,
                "generated_upper_tail_dependence": generated_upper,
                "upper_tail_absolute_error": abs(real_upper - generated_upper),
            })

    return pd.DataFrame(rows, columns=TAIL_DEPENDENCE_COLUMNS)


def _tail_group_summary(
    pairwise_table: pd.DataFrame,
    group_name: str,
) -> dict[str, Any]:
    lower_real = pairwise_table[
        "real_lower_tail_dependence"
    ].to_numpy(dtype=float)
    lower_generated = pairwise_table[
        "generated_lower_tail_dependence"
    ].to_numpy(dtype=float)
    lower_error = pairwise_table[
        "lower_tail_absolute_error"
    ].to_numpy(dtype=float)

    upper_real = pairwise_table[
        "real_upper_tail_dependence"
    ].to_numpy(dtype=float)
    upper_generated = pairwise_table[
        "generated_upper_tail_dependence"
    ].to_numpy(dtype=float)
    upper_error = pairwise_table[
        "upper_tail_absolute_error"
    ].to_numpy(dtype=float)

    return {
        f"tail_num_{group_name}_pairs": int(len(pairwise_table)),
        f"tail_lower_{group_name}_real_mean": _safe_mean(lower_real),
        f"tail_lower_{group_name}_generated_mean": _safe_mean(lower_generated),
        f"tail_lower_{group_name}_mean_abs_error": _safe_mean(lower_error),
        f"tail_lower_{group_name}_median_abs_error": _safe_median(lower_error),
        f"tail_lower_{group_name}_max_abs_error": _safe_max(lower_error),
        f"tail_upper_{group_name}_real_mean": _safe_mean(upper_real),
        f"tail_upper_{group_name}_generated_mean": _safe_mean(upper_generated),
        f"tail_upper_{group_name}_mean_abs_error": _safe_mean(upper_error),
        f"tail_upper_{group_name}_median_abs_error": _safe_median(upper_error),
        f"tail_upper_{group_name}_max_abs_error": _safe_max(upper_error),
    }


def summarize_tail_dependence(
    pairwise_table: pd.DataFrame,
) -> dict[str, Any]:
    """
    Produce benchmark-friendly flat summaries for all, edge, and non-edge pairs.

    The all-pairs summary is always returned. Edge and non-edge summaries are
    populated when tail_dependence_table() was called with an adjacency matrix.
    If graph labels are unavailable, the edge and non-edge values are returned
    as empty-group summaries with None-valued statistics.
    """
    required_columns = {
        "tail_probability",
        "is_graph_edge_pair",
        "real_lower_tail_dependence",
        "generated_lower_tail_dependence",
        "lower_tail_absolute_error",
        "real_upper_tail_dependence",
        "generated_upper_tail_dependence",
        "upper_tail_absolute_error",
    }

    missing = sorted(required_columns.difference(pairwise_table.columns))
    if missing:
        raise ValueError(
            f"Tail-dependence table is missing required columns: {missing}"
        )

    if pairwise_table.empty:
        tail_probability = None
    else:
        unique_probabilities = pairwise_table["tail_probability"].dropna().unique()
        if len(unique_probabilities) != 1:
            raise ValueError(
                "Tail-dependence table must contain exactly one tail_probability."
            )
        tail_probability = float(unique_probabilities[0])

    summary: dict[str, Any] = {
        "tail_probability": tail_probability,
    }

    summary.update(_tail_group_summary(pairwise_table, "all"))

    if pairwise_table["is_graph_edge_pair"].notna().any():
        edge = pairwise_table[pairwise_table["is_graph_edge_pair"] == 1]
        nonedge = pairwise_table[pairwise_table["is_graph_edge_pair"] == 0]
    else:
        edge = pairwise_table.iloc[0:0]
        nonedge = pairwise_table.iloc[0:0]

    summary.update(_tail_group_summary(edge, "edge"))
    summary.update(_tail_group_summary(nonedge, "nonedge"))

    edge_lower_generated = summary["tail_lower_edge_generated_mean"]
    nonedge_lower_generated = summary["tail_lower_nonedge_generated_mean"]
    edge_upper_generated = summary["tail_upper_edge_generated_mean"]
    nonedge_upper_generated = summary["tail_upper_nonedge_generated_mean"]

    summary["tail_lower_generated_edge_nonedge_gap"] = (
        None
        if edge_lower_generated is None or nonedge_lower_generated is None
        else edge_lower_generated - nonedge_lower_generated
    )

    summary["tail_upper_generated_edge_nonedge_gap"] = (
        None
        if edge_upper_generated is None or nonedge_upper_generated is None
        else edge_upper_generated - nonedge_upper_generated
    )

    return summary


def _sanitize_for_json(obj: Any) -> Any:
    """
    Convert numpy types and NaN values to JSON-safe objects.
    """
    if isinstance(obj, dict):
        return {key: _sanitize_for_json(value) for key, value in obj.items()}

    if isinstance(obj, list):
        return [_sanitize_for_json(value) for value in obj]

    if isinstance(obj, tuple):
        return [_sanitize_for_json(value) for value in obj]

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        value = float(obj)
        if np.isnan(value) or np.isinf(value):
            return None
        return value

    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj

    return obj


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)


def save_json(payload: dict | list, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = _sanitize_for_json(payload)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
