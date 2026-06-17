from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
import pickle
import re
from typing import Any

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
    value = _safe_float(getattr(model, "npars", None))
    return 0.0 if value is None else value


def _model_loglik(model: Any, data: np.ndarray) -> float:
    value = _safe_metric(model, "loglik", data)
    return 0.0 if value is None else value


def _clip_u(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), eps, 1.0 - eps)


def build_bicop_fit_controls(
    selection_criterion: str = "bic",
    allow_rotations: bool = True,
    num_threads: int = 1,
):
    # Use names and getattr to avoid static analysis attribute access issues
    family_names = [
        "INDEP",
        "GAUSSIAN",
        "STUDENT",
        "CLAYTON",
        "GUMBEL",
        "FRANK",
    ]

    family_set = []
    for name in family_names:
        fam = getattr(pv.BicopFamily, name, None)
        if fam is not None:
            family_set.append(fam)

    controls = pv.FitControlsBicop(
        family_set=family_set,
        selection_criterion=selection_criterion,
        allow_rotations=allow_rotations,
        num_threads=num_threads,
    )

    family_set_names = family_names
    return controls, family_set_names


def _fit_bicop(data: np.ndarray, controls: Any):
    data = np.asfortranarray(data)
    return pv.Bicop.from_data(
        data,
        controls=controls,
        var_types=["c", "c"],
    )


def _save_bicop(model: Any, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.to_file(str(path))
    return str(path)


def _bicop_summary(model: Any) -> dict:
    return {
        "family": _enum_to_string(model.family),
        "rotation": int(model.rotation) if hasattr(model, "rotation") else None,
        "parameters": _parameters_to_list(model.parameters),
        "npars": _safe_float(getattr(model, "npars", None)),
        "tau": _safe_float(getattr(model, "tau", None)),
    }


@dataclass
class ConditionalVineRecord:
    dataset_id: str
    child: str
    parent_1: str
    parent_2: str
    variables: list[str]
    parent_set_size: int
    n_obs: int
    n_variables: int

    selection_criterion: str
    candidate_families: list[str]

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
            "records": [asdict(record) for record in self.records],
            "implementation": "explicit_3d_dvine_bicop_hfunctions",
        }

        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @staticmethod
    def load_pickle(path: str | Path) -> dict:
        path = Path(path)
        with open(path, "rb") as f:
            return pickle.load(f)


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
) -> ConditionalVineRecord:
    """
    Fit an explicit 3D D-vine local mechanism for:

        child | parent_1, parent_2

    D-vine order:
        parent_1 -- parent_2 -- child

    Fitted bivariate blocks:
        C12    = C(parent_1, parent_2)
        C23    = C(parent_2, child)
        C13|2  = C(F(parent_1|parent_2), F(child|parent_2))
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    variables = [parent_1, parent_2, child]

    u1 = _clip_u(u_df[parent_1].to_numpy(dtype=float), eps=clip_eps)
    u2 = _clip_u(u_df[parent_2].to_numpy(dtype=float), eps=clip_eps)
    uy = _clip_u(u_df[child].to_numpy(dtype=float), eps=clip_eps)

    model_name = (
        f"child_{_safe_filename(child)}"
        f"__given__{_safe_filename(parent_1)}"
        f"__{_safe_filename(parent_2)}"
    )

    c12_path = model_dir / f"{model_name}__c_parent1_parent2.json"
    c23_path = model_dir / f"{model_name}__c_parent2_child.json"
    c13_given_2_path = model_dir / f"{model_name}__c_parent1_child_given_parent2.json"

    try:
        data12 = np.asfortranarray(np.column_stack([u1, u2]))
        data23 = np.asfortranarray(np.column_stack([u2, uy]))

        c12 = _fit_bicop(data12, controls=controls)
        c23 = _fit_bicop(data23, controls=controls)

        # W1 = F(parent_1 | parent_2)
        w1 = c12.hfunc2(data12)

        # WY = F(child | parent_2)
        wy = c23.hfunc1(data23)

        w1 = _clip_u(np.asarray(w1, dtype=float), eps=clip_eps)
        wy = _clip_u(np.asarray(wy, dtype=float), eps=clip_eps)

        data13_given_2 = np.asfortranarray(np.column_stack([w1, wy]))
        c13_given_2 = _fit_bicop(data13_given_2, controls=controls)

        c12_file = _save_bicop(c12, c12_path)
        c23_file = _save_bicop(c23, c23_path)
        c13_given_2_file = _save_bicop(c13_given_2, c13_given_2_path)

        s12 = _bicop_summary(c12)
        s23 = _bicop_summary(c23)
        s13 = _bicop_summary(c13_given_2)

        loglik = (
            _model_loglik(c12, data12)
            + _model_loglik(c23, data23)
            + _model_loglik(c13_given_2, data13_given_2)
        )

        npars = (
            _model_npars(c12)
            + _model_npars(c23)
            + _model_npars(c13_given_2)
        )

        n_obs = len(u_df)
        aic = -2.0 * loglik + 2.0 * npars
        bic = -2.0 * loglik + np.log(n_obs) * npars

        pair_family_summary = (
            f"C({parent_1},{parent_2})={s12['family']}; "
            f"C({parent_2},{child})={s23['family']}; "
            f"C({parent_1},{child}|{parent_2})={s13['family']}"
        )

        return ConditionalVineRecord(
            dataset_id=dataset_id,
            child=child,
            parent_1=parent_1,
            parent_2=parent_2,
            variables=variables,
            parent_set_size=2,
            n_obs=n_obs,
            n_variables=3,
            selection_criterion=selection_criterion,
            candidate_families=candidate_families,
            model_file=c13_given_2_file,
            model_format="explicit_3d_dvine_bicop_json",
            bicop_parent_1_parent_2_file=c12_file,
            bicop_parent_2_child_file=c23_file,
            bicop_parent_1_child_given_parent_2_file=c13_given_2_file,
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
            vine_structure=f"explicit D-vine: {parent_1} -- {parent_2} -- {child}",
            status="ok",
            error=None,
        )

    except Exception as exc:
        return ConditionalVineRecord(
            dataset_id=dataset_id,
            child=child,
            parent_1=parent_1,
            parent_2=parent_2,
            variables=variables,
            parent_set_size=2,
            n_obs=len(u_df),
            n_variables=3,
            selection_criterion=selection_criterion,
            candidate_families=candidate_families,
            model_file=str(c13_given_2_path),
            model_format="explicit_3d_dvine_bicop_json",
            bicop_parent_1_parent_2_file=str(c12_path),
            bicop_parent_2_child_file=str(c23_path),
            bicop_parent_1_child_given_parent_2_file=str(c13_given_2_path),
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
            error=str(exc),
        )


def build_conditional_vine_library(
    u_df: pd.DataFrame,
    dataset_id: str,
    parent_set_size: int = 2,
    selection_criterion: str = "bic",
    allow_rotations: bool = True,
    num_threads: int = 1,
    model_dir: str | Path = "artifacts/conditional_vines_indegree2",
    fixed_variable_order: bool = True,
    limit: int | None = None,
) -> ConditionalVineLibrary:
    """
    Fit explicit 3D D-vine conditional mechanisms for all child-parent-parent triples.

    For d variables, this fits:

        d * C(d - 1, 2)

    local mechanisms.
    """
    if parent_set_size != 2:
        raise NotImplementedError(
            "This builder currently supports parent_set_size=2 only."
        )

    if u_df.empty:
        raise ValueError("Cannot build conditional vine library from empty data.")

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
            f"Fitting explicit D-vine: {child} | {parent_1}, {parent_2}"
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
        )

        records.append(record)

    return ConditionalVineLibrary(
        dataset_id=dataset_id,
        columns=columns,
        n_rows=len(u_df),
        parent_set_size=parent_set_size,
        selection_criterion=selection_criterion,
        family_set=family_set_names,
        records=records,
    )