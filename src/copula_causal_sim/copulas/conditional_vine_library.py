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
        "    pip install pyvinecopulib\n\n"
        "If pip fails on Windows, use conda:\n\n"
        "    conda install -c conda-forge pyvinecopulib\n"
    ) from exc


def _safe_filename(text: str) -> str:
    """
    Create a filesystem-safe name from a variable name.
    """
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_")


def _enum_to_string(value: Any) -> str:
    """
    Convert pyvinecopulib enum values to compact strings.
    """
    if hasattr(value, "name"):
        return str(value.name)

    text = str(value)
    return text.split(".")[-1]


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
    """
    Call pyvinecopulib model metrics robustly.

    For pyvinecopulib, loglik/aic/bic usually accept data.
    This helper also tries no-argument calls for compatibility.
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


def _safe_attr_float(model: Any, attr_name: str) -> float | None:
    try:
        return _safe_float(getattr(model, attr_name))
    except Exception:
        return None


def _safe_string(value: Any) -> str | None:
    try:
        if value is None:
            return None
        return str(value)
    except Exception:
        return None


def _extract_pair_family_summary(model: Any) -> str | None:
    """
    Extract a compact summary of pair-copula families inside a vine model.

    This is intentionally robust because pyvinecopulib object attributes can differ
    across versions.
    """
    candidates = ["pair_copulas", "families"]

    for attr in candidates:
        try:
            value = getattr(model, attr)

            if callable(value):
                value = value()

            return str(value)
        except Exception:
            continue

    return None


def _extract_structure_summary(model: Any) -> str | None:
    """
    Extract a compact structure summary if available.
    """
    try:
        return str(model.structure)
    except Exception:
        return None


def _try_make_fixed_order_structure(num_variables: int):
    """
    Try to create an R-vine structure with fixed variable order.

    For pyvinecopulib versions where this is unavailable or incompatible,
    the caller falls back to automatic structure selection.
    """
    try:
        # vinecopulib typically uses one-based variable labels in structure order.
        order = list(range(1, num_variables + 1))
        return pv.RVineStructure.from_order(order)
    except Exception:
        return None


def _resolve_bicop_family(family_name: str) -> Any:
    """
    Resolve a pyvinecopulib bicopula family enum by name.

    This is done dynamically to support pyvinecopulib versions where enum
    members are not available as lowercase attributes.
    """
    candidates = [family_name, family_name.upper(), family_name.capitalize()]

    for candidate in candidates:
        try:
            return getattr(pv.BicopFamily, candidate)
        except Exception:
            pass

        # try:
        #     return pv.BicopFamily.__members__[candidate]
        # except Exception:
        #     pass

    raise AttributeError(f"Cannot resolve pyvinecopulib bicopula family '{family_name}'")


def build_vine_fit_controls(
    selection_criterion: str = "bic",
    allow_rotations: bool = True,
    num_threads: int = 1,
):
    """
    Create fitting controls for vine copulas.

    Candidate bivariate families:
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
    family_set = [_resolve_bicop_family(name) for name in family_names]

    controls = pv.FitControlsVinecop(
        family_set=family_set,
        selection_criterion=selection_criterion,
        allow_rotations=allow_rotations,
        num_threads=num_threads,
    )

    family_set_names = [_enum_to_string(family) for family in family_set]

    return controls, family_set_names


def _fit_vine_from_data(
    data: np.ndarray,
    controls: Any,
    fixed_variable_order: bool = True,
):
    """
    Fit a vine copula model.

    We first try a fixed variable order. This is useful because we store local
    mechanisms as [parent_1, parent_2, child]. If that is not supported by the
    installed pyvinecopulib version, we fall back to automatic structure selection.
    """
    data = np.asfortranarray(data)
    var_types = ["c"] * data.shape[1]

    if fixed_variable_order:
        structure = _try_make_fixed_order_structure(data.shape[1])

        if structure is not None:
            try:
                return pv.Vinecop.from_data(
                    data,
                    structure=structure,
                    controls=controls,
                    var_types=var_types,
                )
            except Exception:
                pass

    return pv.Vinecop.from_data(
        data,
        controls=controls,
        var_types=var_types,
    )


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
    fixed_variable_order_requested: bool

    model_file: str
    model_format: str

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

    vine_structure: str | None
    pair_family_summary: str | None

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
    fixed_variable_order_requested: bool
    records: list[ConditionalVineRecord]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(record) for record in self.records])

    def save_pickle(self, path: str | Path) -> None:
        """
        Save lightweight metadata and records.

        Individual pyvinecopulib models are saved separately as JSON files.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "dataset_id": self.dataset_id,
            "columns": self.columns,
            "n_rows": self.n_rows,
            "parent_set_size": self.parent_set_size,
            "selection_criterion": self.selection_criterion,
            "family_set": self.family_set,
            "fixed_variable_order_requested": self.fixed_variable_order_requested,
            "records": [asdict(record) for record in self.records],
        }

        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @staticmethod
    def load_pickle(path: str | Path) -> dict:
        path = Path(path)

        with open(path, "rb") as f:
            return pickle.load(f)


def _save_vine_model(model: Any, model_path: Path) -> tuple[str, str]:
    """
    Save a fitted vine model.

    Primary format: pyvinecopulib JSON using model.to_file().
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)

    json_path = model_path.with_suffix(".json")

    model.to_file(str(json_path))

    return str(json_path), "json"


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
    fixed_variable_order: bool = True,
) -> ConditionalVineRecord:
    """
    Fit one 3-variable vine model for:

        child | parent_1, parent_2

    The stored variable order is:

        [parent_1, parent_2, child]
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    variables = [parent_1, parent_2, child]
    data = u_df[variables].to_numpy(dtype=float)

    model_name = (
        f"child_{_safe_filename(child)}"
        f"__given__{_safe_filename(parent_1)}"
        f"__{_safe_filename(parent_2)}"
    )

    model_base_path = model_dir / model_name

    try:
        model = _fit_vine_from_data(
            data=data,
            controls=controls,
            fixed_variable_order=fixed_variable_order,
        )

        model_file, model_format = _save_vine_model(model, model_base_path)

        p1 = data[:, 0]
        p2 = data[:, 1]
        y = data[:, 2]

        record = ConditionalVineRecord(
            dataset_id=dataset_id,
            child=child,
            parent_1=parent_1,
            parent_2=parent_2,
            variables=variables,
            parent_set_size=2,
            n_obs=data.shape[0],
            n_variables=data.shape[1],
            selection_criterion=selection_criterion,
            candidate_families=candidate_families,
            fixed_variable_order_requested=fixed_variable_order,
            model_file=model_file,
            model_format=model_format,
            npars=_safe_attr_float(model, "npars"),
            loglik=_safe_metric(model, "loglik", data),
            aic=_safe_metric(model, "aic", data),
            bic=_safe_metric(model, "bic", data),
            kendall_parent_1_child=_safe_float(kendalltau(p1, y)[0]),
            kendall_parent_2_child=_safe_float(kendalltau(p2, y)[0]),
            kendall_parent_1_parent_2=_safe_float(kendalltau(p1, p2)[0]),
            spearman_parent_1_child=_safe_float(spearmanr(p1, y)[0]),
            spearman_parent_2_child=_safe_float(spearmanr(p2, y)[0]),
            spearman_parent_1_parent_2=_safe_float(spearmanr(p1, p2)[0]),
            vine_structure=_extract_structure_summary(model),
            pair_family_summary=_extract_pair_family_summary(model),
            status="ok",
            error=None,
        )

        return record

    except Exception as exc:
        return ConditionalVineRecord(
            dataset_id=dataset_id,
            child=child,
            parent_1=parent_1,
            parent_2=parent_2,
            variables=variables,
            parent_set_size=2,
            n_obs=data.shape[0],
            n_variables=data.shape[1],
            selection_criterion=selection_criterion,
            candidate_families=candidate_families,
            fixed_variable_order_requested=fixed_variable_order,
            model_file=str(model_base_path.with_suffix(".json")),
            model_format="json",
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
            vine_structure=None,
            pair_family_summary=None,
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
    Fit conditional vine models for all child-parent-parent triples.

    Currently implemented:
        parent_set_size = 2

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

    controls, family_set_names = build_vine_fit_controls(
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
            f"Fitting conditional vine: {child} | {parent_1}, {parent_2}"
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
            fixed_variable_order=fixed_variable_order,
        )

        records.append(record)

    return ConditionalVineLibrary(
        dataset_id=dataset_id,
        columns=columns,
        n_rows=len(u_df),
        parent_set_size=parent_set_size,
        selection_criterion=selection_criterion,
        family_set=family_set_names,
        fixed_variable_order_requested=fixed_variable_order,
        records=records,
    )