from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
import pandas as pd


def make_pseudo_observations(
    df: pd.DataFrame,
    clip_eps: float = 1e-6,
) -> pd.DataFrame:
    """
    Transform each column into pseudo-observations in (0, 1).

    Uses average ranks:

        U_ij = rank(X_ij) / (n + 1)

    The output is clipped to [clip_eps, 1 - clip_eps] for numerical stability.
    """
    if df.empty:
        raise ValueError("Cannot create pseudo-observations from an empty dataframe.")

    n = len(df)

    u = df.rank(method="average") / (n + 1.0)
    u = u.clip(lower=clip_eps, upper=1.0 - clip_eps)

    return u


@dataclass
class EmpiricalMarginal:
    """
    Empirical marginal model for one variable.

    Stores sorted observed values and uses empirical quantiles for inverse transform.
    """
    name: str
    sorted_values: np.ndarray
    n: int
    min_value: float
    max_value: float

    @classmethod
    def fit(cls, name: str, values: np.ndarray) -> "EmpiricalMarginal":
        values = np.asarray(values, dtype=float)

        if values.ndim != 1:
            raise ValueError(f"Expected 1D values for {name}, got shape {values.shape}.")

        if len(values) == 0:
            raise ValueError(f"Cannot fit marginal for {name}: empty values.")

        if np.isnan(values).any():
            raise ValueError(f"Cannot fit marginal for {name}: NaN values found.")

        sorted_values = np.sort(values)

        return cls(
            name=name,
            sorted_values=sorted_values,
            n=len(sorted_values),
            min_value=float(sorted_values[0]),
            max_value=float(sorted_values[-1]),
        )

    def inverse_cdf(self, u: np.ndarray, clip_eps: float = 1e-6) -> np.ndarray:
        """
        Map U values in (0, 1) back to the original data scale.

        Uses linear interpolation through numpy.quantile.
        """
        u = np.asarray(u, dtype=float)
        u = np.clip(u, clip_eps, 1.0 - clip_eps)

        return np.quantile(
            self.sorted_values,
            u,
            method="linear",
        )

@dataclass
class EmpiricalMarginalLibrary:
    """
    Collection of empirical marginal models for all variables.
    """
    dataset_id: str
    columns: list[str]
    n_rows: int
    clip_eps: float
    marginals: dict[str, EmpiricalMarginal]

    @classmethod
    def fit(
        cls,
        df: pd.DataFrame,
        dataset_id: str = "sachs",
        clip_eps: float = 1e-6,
    ) -> "EmpiricalMarginalLibrary":
        marginals = {}

        for col in df.columns:
            marginals[col] = EmpiricalMarginal.fit(
                name=col,
                values=df[col].to_numpy(),
            )

        return cls(
            dataset_id=dataset_id,
            columns=list(df.columns),
            n_rows=len(df),
            clip_eps=clip_eps,
            marginals=marginals,
        )

    def inverse_transform(self, u_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform pseudo-observations or synthetic copula samples back to original scale.
        """
        if list(u_df.columns) != self.columns:
            raise ValueError(
                "Column mismatch between input U dataframe and marginal library.\n"
                f"Expected: {self.columns}\n"
                f"Got: {list(u_df.columns)}"
            )

        x = {}

        for col in self.columns:
            x[col] = self.marginals[col].inverse_cdf(
                u_df[col].to_numpy(),
                clip_eps=self.clip_eps,
            )

        return pd.DataFrame(x, columns=self.columns)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "EmpiricalMarginalLibrary":
        path = Path(path)

        with open(path, "rb") as f:
            return pickle.load(f)