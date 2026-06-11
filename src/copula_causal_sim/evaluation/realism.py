from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Literal

import numpy as np
import pandas as pd


DEFAULT_QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]


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


def correlation_matrix(df: pd.DataFrame, method: Literal["kendall", "spearman", "pearson"]) -> pd.DataFrame:
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

    return float(np.nanmean(values))


def _safe_median(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None

    return float(np.nanmedian(values))


def _safe_max(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None

    return float(np.nanmax(values))


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