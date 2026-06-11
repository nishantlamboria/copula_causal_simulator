from __future__ import annotations

from pathlib import Path
import json
from typing import Literal

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


def compute_marginal_quantile_errors(
    real_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    quantiles: list[float] | None = None,
) -> pd.DataFrame:
    """
    Compare marginal empirical quantiles between real and generated data.

    Returns a long-format table:
        variable, quantile, real_value, generated_value, absolute_error
    """
    validate_same_columns(real_df, generated_df)

    if quantiles is None:
        quantiles = DEFAULT_QUANTILES

    rows = []

    for col in real_df.columns:
        for q in quantiles:
            real_value = float(real_df[col].quantile(q))
            generated_value = float(generated_df[col].quantile(q))

            rows.append({
                "variable": col,
                "quantile": q,
                "real_value": real_value,
                "generated_value": generated_value,
                "absolute_error": abs(real_value - generated_value),
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
            max_abs_quantile_error=("absolute_error", "max"),
        )
        .sort_values("mean_abs_quantile_error", ascending=False)
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


def summarize_matrix_error(error_matrix: pd.DataFrame) -> dict:
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

            real_value = float(np.array(real_matrix.at[var1, var2]).item())
            generated_value = float(np.array(generated_matrix.at[var1, var2]).item())
            absolute_error = float(np.array(error_matrix.at[var1, var2]).item())

            rows.append({
                "var1": var1,
                "var2": var2,
                "real_value": real_value,
                "generated_value": generated_value,
                "absolute_error": absolute_error,
            })

    return (
        pd.DataFrame(rows)
        .sort_values("absolute_error", ascending=False)
        .head(top_k)
    )


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)


def save_json(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)