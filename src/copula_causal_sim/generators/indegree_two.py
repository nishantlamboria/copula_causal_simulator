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
        "pyvinecopulib is required for conditional vine generation."
    ) from exc

from copula_causal_sim.copulas.marginals import EmpiricalMarginalLibrary
from copula_causal_sim.generators.indegree_one import (
    IndegreeOneGenerator,
    load_paircopula_metadata,
)
from copula_causal_sim.graphs.dag import (
    parents_of,
    topological_order,
    validate_dag,
)


@dataclass
class IndegreeTwoGenerationResult:
    u: pd.DataFrame
    x: pd.DataFrame
    adjacency: np.ndarray
    columns: list[str]
    seed: int
    max_supported_indegree: int = 2


def load_conditional_vine_metadata(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Conditional vine library metadata not found: {path}")

    with open(path, "rb") as f:
        payload = pickle.load(f)

    required = [
        "dataset_id",
        "columns",
        "parent_set_size",
        "records",
    ]

    missing = [key for key in required if key not in payload]

    if missing:
        raise ValueError(
            f"Invalid conditional vine metadata. Missing keys: {missing}"
        )

    if int(payload["parent_set_size"]) != 2:
        raise ValueError(
            "This generator requires a conditional vine library with parent_set_size=2."
        )

    return payload


class IndegreeTwoGenerator:
    """
    Graph-conditioned generator for DAGs with max indegree <= 2.

    Generation rules:

        root node:
            U_j ~ Uniform(0, 1)

        one-parent node:
            U_child ~ C_child|parent(. | U_parent)
            using fitted bivariate pair copula

        two-parent node:
            U_child ~ C_child|parent1,parent2(. | U_parent1, U_parent2)
            using fitted 3D vine copula

    After generating all U variables, empirical inverse marginals map U back
    to the original data scale.
    """

    def __init__(
        self,
        marginal_library: EmpiricalMarginalLibrary,
        paircopula_metadata: dict[str, Any],
        conditional_vine_metadata: dict[str, Any],
        artifact_dir: str | Path,
        clip_eps: float = 1e-6,
    ) -> None:
        self.marginal_library = marginal_library
        self.paircopula_metadata = paircopula_metadata
        self.conditional_vine_metadata = conditional_vine_metadata
        self.artifact_dir = Path(artifact_dir)
        self.clip_eps = float(clip_eps)

        self.columns = list(marginal_library.columns)
        self.column_to_index = {col: idx for idx, col in enumerate(self.columns)}

        pair_columns = list(paircopula_metadata["columns"])
        vine_columns = list(conditional_vine_metadata["columns"])

        if pair_columns != self.columns:
            raise ValueError(
                "Column mismatch between marginal library and pair-copula metadata.\n"
                f"Marginals: {self.columns}\n"
                f"Pair copulas: {pair_columns}"
            )

        if vine_columns != self.columns:
            raise ValueError(
                "Column mismatch between marginal library and conditional vine metadata.\n"
                f"Marginals: {self.columns}\n"
                f"Conditional vines: {vine_columns}"
            )

        self.one_parent_generator = IndegreeOneGenerator(
            marginal_library=marginal_library,
            paircopula_metadata=paircopula_metadata,
            artifact_dir=artifact_dir,
            clip_eps=clip_eps,
        )

        self.conditional_records = conditional_vine_metadata["records"]
        self.conditional_lookup = self._build_conditional_lookup(
            self.conditional_records
        )

        self.vine_model_cache: dict[str, Any] = {}

    @staticmethod
    def _build_conditional_lookup(
        records: list[dict[str, Any]],
    ) -> dict[tuple[str, frozenset[str]], dict[str, Any]]:
        """
        Build lookup:

            (child, frozenset({parent_1, parent_2})) -> record

        The parent set is unordered for graph lookup. The stored record still
        preserves the fitted model order [parent_1, parent_2, child].
        """
        lookup: dict[tuple[str, frozenset[str]], dict[str, Any]] = {}

        for record in records:
            if record.get("status") != "ok":
                continue

            child = record["child"]
            parent_set = frozenset([record["parent_1"], record["parent_2"]])

            lookup[(child, parent_set)] = record

        return lookup

    def _get_conditional_record(
        self,
        child_name: str,
        parent_names: list[str],
    ) -> dict[str, Any]:
        if len(parent_names) != 2:
            raise ValueError(
                f"Expected exactly two parents, got {len(parent_names)}."
            )

        key = (child_name, frozenset(parent_names))

        if key not in self.conditional_lookup:
            raise KeyError(
                "No fitted conditional vine found for local mechanism:\n"
                f"child={child_name}, parents={parent_names}"
            )

        return self.conditional_lookup[key]

    def _resolve_vine_model_path(self, record: dict[str, Any]) -> Path:
        stored_path = Path(record["model_file"])

        if stored_path.exists():
            return stored_path

        fallback_path = (
            self.artifact_dir
            / "conditional_vines_indegree2"
            / stored_path.name
        )

        if fallback_path.exists():
            return fallback_path

        raise FileNotFoundError(
            "Could not find fitted conditional vine model file.\n"
            f"Stored path: {stored_path}\n"
            f"Fallback path: {fallback_path}"
        )

    def _load_vine_model(self, record: dict[str, Any]):
        model_path = self._resolve_vine_model_path(record)
        cache_key = str(model_path)

        if cache_key not in self.vine_model_cache:
            self.vine_model_cache[cache_key] = pv.Vinecop.from_file(
                str(model_path)
            )

        return self.vine_model_cache[cache_key]

    def _sample_child_given_two_parents(
        self,
        child_name: str,
        parent_names: list[str],
        u_matrix: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Sample U_child | U_parent1, U_parent2 using a fitted 3D vine.

        Each conditional vine was fitted on variables in the stored order:

            [record["parent_1"], record["parent_2"], record["child"]]

        For conditional simulation, we:
        1. take the already generated parent values,
        2. compute the first two Rosenblatt coordinates for the parent pair,
        3. draw a fresh uniform coordinate for the child,
        4. apply inverse Rosenblatt,
        5. return the third component as U_child.

        This relies on the fixed variable order used during fitting.
        """
        record = self._get_conditional_record(
            child_name=child_name,
            parent_names=parent_names,
        )

        model = self._load_vine_model(record)

        record_parent_1 = record["parent_1"]
        record_parent_2 = record["parent_2"]
        record_child = record["child"]

        if record_child != child_name:
            raise RuntimeError(
                "Conditional vine record has unexpected child.\n"
                f"Expected: {child_name}\n"
                f"Got: {record_child}"
            )

        parent_1_idx = self.column_to_index[record_parent_1]
        parent_2_idx = self.column_to_index[record_parent_2]

        u_parent_1 = u_matrix[:, parent_1_idx]
        u_parent_2 = u_matrix[:, parent_2_idx]

        n_samples = len(u_parent_1)

        # Dummy child value. In the fixed order [P1, P2, Y], the first two
        # Rosenblatt coordinates depend only on P1 and P2.
        dummy_child = np.full(n_samples, 0.5, dtype=float)

        observed_parent_data = np.column_stack(
            [u_parent_1, u_parent_2, dummy_child]
        )
        observed_parent_data = np.asfortranarray(observed_parent_data)

        rosenblatt_values = model.rosenblatt(observed_parent_data)

        q_child = rng.uniform(
            self.clip_eps,
            1.0 - self.clip_eps,
            size=n_samples,
        )

        conditional_rosenblatt_values = np.array(
            rosenblatt_values,
            dtype=float,
            copy=True,
            order="F",
        )

        conditional_rosenblatt_values[:, 2] = q_child
        conditional_rosenblatt_values = np.asfortranarray(
            conditional_rosenblatt_values
        )

        simulated = model.inverse_rosenblatt(conditional_rosenblatt_values)

        u_child = np.asarray(simulated[:, 2], dtype=float)
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

        adj = validate_dag(adjacency, max_allowed_indegree=2)

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
            parent_names = [self.columns[parent] for parent in parents]

            if len(parents) == 0:
                u[:, node] = rng.uniform(
                    self.clip_eps,
                    1.0 - self.clip_eps,
                    size=n_samples,
                )

            elif len(parents) == 1:
                parent = parents[0]
                parent_name = self.columns[parent]

                u[:, node] = self.one_parent_generator._sample_child_given_parent(
                    parent_name=parent_name,
                    child_name=node_name,
                    u_parent=u[:, parent],
                    rng=rng,
                )

            elif len(parents) == 2:
                u[:, node] = self._sample_child_given_two_parents(
                    child_name=node_name,
                    parent_names=parent_names,
                    u_matrix=u,
                    rng=rng,
                )

            else:
                raise ValueError(
                    f"Node {node_name} has {len(parents)} parents. "
                    "This generator supports max indegree <= 2."
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
    ) -> IndegreeTwoGenerationResult:
        u_df = self.sample_u(
            adjacency=adjacency,
            n_samples=n_samples,
            seed=seed,
        )

        x_df = self.marginal_library.inverse_transform(u_df)

        return IndegreeTwoGenerationResult(
            u=u_df,
            x=x_df,
            adjacency=adjacency,
            columns=self.columns,
            seed=seed,
            max_supported_indegree=2,
        )