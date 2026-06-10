from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pandas as pd

try:
    import pyvinecopulib as pv
except ImportError as exc:
    raise ImportError(
        "pyvinecopulib is required for pair-copula conditional generation."
    ) from exc

from copula_causal_sim.copulas.marginals import EmpiricalMarginalLibrary
from copula_causal_sim.graphs.dag import (
    parents_of,
    topological_order,
    validate_dag,
)


@dataclass
class IndegreeOneGenerationResult:
    u: pd.DataFrame
    x: pd.DataFrame
    adjacency: np.ndarray
    columns: list[str]
    seed: int


def load_paircopula_metadata(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Pair-copula library metadata not found: {path}")

    with open(path, "rb") as f:
        payload = pickle.load(f)

    required = ["dataset_id", "columns", "records"]

    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Invalid pair-copula metadata. Missing keys: {missing}")

    return payload


class IndegreeOneGenerator:
    """
    Graph-conditioned generator for DAGs with max indegree <= 1.

    The generator works in copula space first:

        root node:
            U_j ~ Uniform(0, 1)

        one-parent node:
            U_child ~ C_child|parent(. | U_parent)

    Then empirical marginal inverse CDFs map U back to the original data scale.
    """

    def __init__(
        self,
        marginal_library: EmpiricalMarginalLibrary,
        paircopula_metadata: dict[str, Any],
        artifact_dir: str | Path,
        clip_eps: float = 1e-6,
    ) -> None:
        self.marginal_library = marginal_library
        self.paircopula_metadata = paircopula_metadata
        self.artifact_dir = Path(artifact_dir)
        self.clip_eps = float(clip_eps)

        self.columns = list(marginal_library.columns)

        metadata_columns = list(paircopula_metadata["columns"])
        if metadata_columns != self.columns:
            raise ValueError(
                "Column mismatch between marginal library and pair-copula metadata.\n"
                f"Marginals: {self.columns}\n"
                f"Pair copulas: {metadata_columns}"
            )

        self.records = paircopula_metadata["records"]
        self.record_lookup = self._build_record_lookup(self.records)
        self.model_cache: dict[str, Any] = {}

    @staticmethod
    def _build_record_lookup(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
        lookup: dict[tuple[str, str], dict[str, Any]] = {}

        for record in records:
            if record.get("status") != "ok":
                continue

            var1 = record["var1"]
            var2 = record["var2"]

            lookup[(var1, var2)] = record
            lookup[(var2, var1)] = record

        return lookup

    def _get_record(self, parent: str, child: str) -> dict[str, Any]:
        key = (parent, child)

        if key not in self.record_lookup:
            raise KeyError(f"No fitted pair copula found for pair: {parent}, {child}")

        return self.record_lookup[key]

    def _resolve_model_path(self, record: dict[str, Any]) -> Path:
        stored_path = Path(record["model_file"])

        if stored_path.exists():
            return stored_path

        fallback_path = self.artifact_dir / "paircopula_models" / stored_path.name

        if fallback_path.exists():
            return fallback_path

        raise FileNotFoundError(
            "Could not find fitted pair-copula model file.\n"
            f"Stored path: {stored_path}\n"
            f"Fallback path: {fallback_path}"
        )

    def _load_model(self, record: dict[str, Any]):
        model_path = self._resolve_model_path(record)
        cache_key = str(model_path)

        if cache_key not in self.model_cache:
            self.model_cache[cache_key] = pv.Bicop.from_file(str(model_path))

        return self.model_cache[cache_key]

    def _sample_child_given_parent(
        self,
        parent_name: str,
        child_name: str,
        u_parent: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Sample U_child | U_parent using inverse h-functions.

        Case 1:
            fitted model order is (parent, child).
            Use hinv1 because hfunc1 gives P(U_child <= u_child | U_parent).

        Case 2:
            fitted model order is (child, parent).
            Use hinv2 because hfunc2 gives P(U_child <= u_child | U_parent).
        """
        record = self._get_record(parent_name, child_name)
        model = self._load_model(record)

        n = len(u_parent)
        q = rng.uniform(self.clip_eps, 1.0 - self.clip_eps, size=n)

        var1 = record["var1"]
        var2 = record["var2"]

        if var1 == parent_name and var2 == child_name:
            data = np.column_stack([u_parent, q])
            data = np.asfortranarray(data)
            u_child = model.hinv1(data)

        elif var1 == child_name and var2 == parent_name:
            data = np.column_stack([q, u_parent])
            data = np.asfortranarray(data)
            u_child = model.hinv2(data)

        else:
            raise RuntimeError(
                "Pair-copula record does not match requested parent-child pair.\n"
                f"Requested: {parent_name} -> {child_name}\n"
                f"Record: {var1}, {var2}"
            )

        u_child = np.asarray(u_child, dtype=float)
        u_child = np.clip(u_child, self.clip_eps, 1.0 - self.clip_eps)

        return u_child

    def sample_u(
        self,
        adjacency: np.ndarray,
        n_samples: int,
        seed: int,
    ) -> pd.DataFrame:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")

        adj = validate_dag(adjacency, max_allowed_indegree=1)

        if adj.shape[0] != len(self.columns):
            raise ValueError(
                "Graph size does not match number of dataset variables.\n"
                f"Graph nodes: {adj.shape[0]}\n"
                f"Dataset variables: {len(self.columns)}"
            )

        rng = np.random.default_rng(seed)

        u = np.full(
            shape=(n_samples, len(self.columns)),
            fill_value=np.nan,
            dtype=float,
        )

        order = topological_order(adj)

        for node in order:
            node_name = self.columns[node]
            parents = parents_of(adj, node)

            if len(parents) == 0:
                u[:, node] = rng.uniform(
                    self.clip_eps,
                    1.0 - self.clip_eps,
                    size=n_samples,
                )

            elif len(parents) == 1:
                parent = parents[0]
                parent_name = self.columns[parent]

                u[:, node] = self._sample_child_given_parent(
                    parent_name=parent_name,
                    child_name=node_name,
                    u_parent=u[:, parent],
                    rng=rng,
                )

            else:
                raise ValueError(
                    f"Node {node_name} has {len(parents)} parents. "
                    "This generator only supports max indegree <= 1."
                )

        if np.isnan(u).any():
            raise RuntimeError("Generated U contains NaN values.")

        u = np.clip(u, self.clip_eps, 1.0 - self.clip_eps)

        return pd.DataFrame(u, columns=self.columns)

    def generate(
        self,
        adjacency: np.ndarray,
        n_samples: int,
        seed: int,
    ) -> IndegreeOneGenerationResult:
        u_df = self.sample_u(
            adjacency=adjacency,
            n_samples=n_samples,
            seed=seed,
        )

        x_df = self.marginal_library.inverse_transform(u_df)

        return IndegreeOneGenerationResult(
            u=u_df,
            x=x_df,
            adjacency=adjacency,
            columns=self.columns,
            seed=seed,
        )