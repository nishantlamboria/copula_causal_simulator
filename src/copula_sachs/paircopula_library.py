from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, kendalltau


try:
    import pyvinecopulib as pv
except ImportError as exc:
    raise ImportError(
        "pyvinecopulib is not installed. Install it with:\n\n"
        "    pip install pyvinecopulib\n\n"
        "If pip fails on Windows, use conda:\n\n"
        "    conda install -c conda-forge pyvinecopulib\n"
    ) from exc


def _enum_to_string(value: Any) -> str:
    """
    Convert pyvinecopulib enum values to compact strings.
    """
    if hasattr(value, "name"):
        return str(value.name)
    text = str(value)
    return text.split(".")[-1]


def _parameters_to_list(parameters: Any) -> list:
    """
    Convert pyvinecopulib parameter matrix to a JSON/CSV-friendly list.
    """
    try:
        arr = np.asarray(parameters, dtype=float)
        return arr.tolist()
    except Exception:
        return []


def _safe_float(value: Any) -> float | None:
    """
    Convert a value to float if possible, otherwise return None.
    """
    try:
        if value is None:
            return None
        value = float(value)
        if np.isnan(value):
            return None
        return value
    except Exception:
        return None


def _safe_metric(model: Any, metric_name: str, data: np.ndarray) -> float | None:
    """
    Call pyvinecopulib metrics robustly.

    Some methods require data as input. This helper tries the common API first.
    """
    try:
        method = getattr(model, metric_name)
        return _safe_float(method(data))
    except Exception:
        try:
            method = getattr(model, metric_name)
            return _safe_float(method())
        except Exception:
            return None


def empirical_tail_dependence(
    u_pair: np.ndarray,
    q: float = 0.05,
) -> tuple[float, float]:
    """
    Simple empirical lower and upper tail dependence estimates.

    lower = P(U1 <= q, U2 <= q) / q
    upper = P(U1 >= 1-q, U2 >= 1-q) / q

    This is only a diagnostic, not a precise asymptotic tail-dependence estimator.
    """
    u1 = u_pair[:, 0]
    u2 = u_pair[:, 1]

    lower = np.mean((u1 <= q) & (u2 <= q)) / q
    upper = np.mean((u1 >= 1.0 - q) & (u2 >= 1.0 - q)) / q

    return float(lower), float(upper)


@dataclass
class PairCopulaRecord:
    dataset_id: str
    var1: str
    var2: str
    family: str
    rotation: int | None
    parameters: list
    npars: float | None
    model_tau: float | None
    empirical_kendall_tau: float | None
    empirical_spearman_rho: float | None
    empirical_lower_tail_q05: float | None
    empirical_upper_tail_q05: float | None
    loglik: float | None
    aic: float | None
    bic: float | None
    n_obs: int
    selection_criterion: str
    model_file: str
    status: str
    error: str | None = None


@dataclass
class PairCopulaLibrary:
    dataset_id: str
    columns: list[str]
    n_rows: int
    selection_criterion: str
    family_set: list[str]
    records: list[PairCopulaRecord]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(record) for record in self.records])

    def save_pickle(self, path: str | Path) -> None:
        """
        Save the lightweight library metadata and records.

        The actual fitted pyvinecopulib Bicop models are saved separately as JSON files.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "dataset_id": self.dataset_id,
            "columns": self.columns,
            "n_rows": self.n_rows,
            "selection_criterion": self.selection_criterion,
            "family_set": self.family_set,
            "records": [asdict(record) for record in self.records],
        }

        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @staticmethod
    def load_pickle(path: str | Path) -> dict:
        path = Path(path)
        with open(path, "rb") as f:
            return pickle.load(f)


def build_fit_controls(
    selection_criterion: str = "bic",
    allow_rotations: bool = True,
    num_threads: int = 1,
):
    """
    Create pyvinecopulib fitting controls for bivariate copulas.

    Candidate families:
    - independence
    - Gaussian
    - Student-t
    - Clayton
    - Gumbel
    - Frank
    """
    family_names = [
            "indep",
            "gaussian",
            "student",
            "clayton",
            "gumbel",
            "frank",
        ]

    family_set = []
    for name in family_names:
        try:
            fam = getattr(pv.BicopFamily, name)
        except Exception:
            # If the attribute is not present, try uppercase variant
            try:
                fam = getattr(pv.BicopFamily, name.upper())
            except Exception:
                continue
        family_set.append(fam)

    controls = pv.FitControlsBicop(
        family_set=family_set,
        selection_criterion=selection_criterion,
        allow_rotations=allow_rotations,
        num_threads=num_threads,
    )

    family_set_names = [_enum_to_string(fam) for fam in family_set]

    return controls, family_set_names


def fit_one_paircopula(
    u_df: pd.DataFrame,
    var1: str,
    var2: str,
    dataset_id: str,
    controls: Any,
    selection_criterion: str,
    model_dir: str | Path,
) -> PairCopulaRecord:
    """
    Fit one selected bivariate copula model for a variable pair.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    pair_data = u_df[[var1, var2]].to_numpy(dtype=float)
    pair_data = np.asfortranarray(pair_data)

    model_filename = f"{var1}__{var2}.json"
    model_path = model_dir / model_filename

    try:
        model = pv.Bicop.from_data(
            pair_data,
            controls=controls,
            var_types=["c", "c"],
        )

        # Save fitted model as pyvinecopulib JSON file.
        # This is safer than trying to pickle the C++/pybind object directly.
        model.to_file(str(model_path))

        empirical_tau, _ = kendalltau(pair_data[:, 0], pair_data[:, 1])
        empirical_rho, _ = spearmanr(pair_data[:, 0], pair_data[:, 1])
        lower_tail, upper_tail = empirical_tail_dependence(pair_data, q=0.05)

        record = PairCopulaRecord(
            dataset_id=dataset_id,
            var1=var1,
            var2=var2,
            family=_enum_to_string(model.family),
            rotation=int(model.rotation) if hasattr(model, "rotation") else None,
            parameters=_parameters_to_list(model.parameters),
            npars=_safe_float(getattr(model, "npars", None)),
            model_tau=_safe_float(getattr(model, "tau", None)),
            empirical_kendall_tau=_safe_float(empirical_tau),
            empirical_spearman_rho=_safe_float(empirical_rho),
            empirical_lower_tail_q05=_safe_float(lower_tail),
            empirical_upper_tail_q05=_safe_float(upper_tail),
            loglik=_safe_metric(model, "loglik", pair_data),
            aic=_safe_metric(model, "aic", pair_data),
            bic=_safe_metric(model, "bic", pair_data),
            n_obs=pair_data.shape[0],
            selection_criterion=selection_criterion,
            model_file=str(model_path),
            status="ok",
            error=None,
        )

        return record

    except Exception as exc:
        return PairCopulaRecord(
            dataset_id=dataset_id,
            var1=var1,
            var2=var2,
            family="fit_failed",
            rotation=None,
            parameters=[],
            npars=None,
            model_tau=None,
            empirical_kendall_tau=None,
            empirical_spearman_rho=None,
            empirical_lower_tail_q05=None,
            empirical_upper_tail_q05=None,
            loglik=None,
            aic=None,
            bic=None,
            n_obs=pair_data.shape[0],
            selection_criterion=selection_criterion,
            model_file=str(model_path),
            status="failed",
            error=str(exc),
        )


def build_paircopula_library(
    u_df: pd.DataFrame,
    dataset_id: str = "sachs",
    selection_criterion: str = "bic",
    allow_rotations: bool = True,
    num_threads: int = 1,
    model_dir: str | Path = "artifacts/paircopula_models_sachs_v1",
) -> PairCopulaLibrary:
    """
    Fit selected pairwise copulas for all variable pairs in u_df.
    """
    controls, family_set_names = build_fit_controls(
        selection_criterion=selection_criterion,
        allow_rotations=allow_rotations,
        num_threads=num_threads,
    )

    columns = list(u_df.columns)
    records: list[PairCopulaRecord] = []

    pairs = list(combinations(columns, 2))
    total = len(pairs)

    for idx, (var1, var2) in enumerate(pairs, start=1):
        print(f"[{idx:02d}/{total:02d}] Fitting pair: {var1} -- {var2}")

        record = fit_one_paircopula(
            u_df=u_df,
            var1=var1,
            var2=var2,
            dataset_id=dataset_id,
            controls=controls,
            selection_criterion=selection_criterion,
            model_dir=model_dir,
        )

        records.append(record)

    return PairCopulaLibrary(
        dataset_id=dataset_id,
        columns=columns,
        n_rows=len(u_df),
        selection_criterion=selection_criterion,
        family_set=family_set_names,
        records=records,
    )