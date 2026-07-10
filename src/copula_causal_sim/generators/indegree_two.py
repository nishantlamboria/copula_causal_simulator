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
            using the selected explicit 3D D-vine parent order:

                selected_parent_1 -- selected_parent_2 -- child

            The DAG parent set is unchanged; only the statistical D-vine
            representation is selected from the two admissible orders.
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

        self.bicop_cache: dict[str, Any] = {}

    @staticmethod
    def _build_conditional_lookup(
        records: list[dict[str, Any]],
    ) -> dict[tuple[str, frozenset[str]], dict[str, Any]]:
        lookup: dict[tuple[str, frozenset[str]], dict[str, Any]] = {}

        for record in records:
            if record.get("status") != "ok":
                continue

            child = record["child"]
            stored_parent_set = record.get(
                "parent_set",
                [record["parent_1"], record["parent_2"]],
            )
            parent_set = frozenset(stored_parent_set)

            key = (child, parent_set)
            if key in lookup:
                raise ValueError(
                    "Duplicate conditional D-vine records for "
                    f"child={child!r}, parents={sorted(parent_set)!r}."
                )

            lookup[key] = record

        return lookup

    @staticmethod
    def _selected_order_from_record(
        record: dict[str, Any],
    ) -> list[str]:
        """Return and validate the stored D-vine order.

        Older fixed-order metadata did not include ``selected_order``; for
        those records the ordered ``parent_1``, ``parent_2``, ``child`` fields
        remain the authoritative representation.
        """
        record_parent_1 = record["parent_1"]
        record_parent_2 = record["parent_2"]
        record_child = record["child"]

        selected_order = record.get(
            "selected_order",
            [record_parent_1, record_parent_2, record_child],
        )

        expected = [record_parent_1, record_parent_2, record_child]
        if (
            not isinstance(selected_order, (list, tuple))
            or len(selected_order) != 3
            or list(selected_order) != expected
        ):
            raise ValueError(
                "Conditional record has inconsistent selected-order metadata. "
                f"Expected {expected!r}, got {selected_order!r}."
            )

        return list(selected_order)

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
                "No fitted conditional D-vine found for local mechanism:\n"
                f"child={child_name}, parents={parent_names}"
            )

        return self.conditional_lookup[key]

    def _resolve_model_path(self, stored_path: str | Path) -> Path:
        stored_path = Path(stored_path)

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
            "Could not find fitted bivariate copula model file.\n"
            f"Stored path: {stored_path}\n"
            f"Fallback path: {fallback_path}"
        )

    def _load_bicop(self, stored_path: str | Path):
        model_path = self._resolve_model_path(stored_path)
        cache_key = str(model_path)

        if cache_key not in self.bicop_cache:
            self.bicop_cache[cache_key] = pv.Bicop.from_file(str(model_path))

        return self.bicop_cache[cache_key]

    def _sample_child_given_two_parents(
        self,
        child_name: str,
        parent_names: list[str],
        u_matrix: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Sample U_child | U_parent1, U_parent2 using explicit D-vine h-functions.

        Stored D-vine order:

            parent_1 -- parent_2 -- child

        Stored copulas:

            C12      = C(parent_1, parent_2)
            C23      = C(parent_2, child)
            C13|2    = C(F(parent_1|parent_2), F(child|parent_2))

        Sampling:

            w1 = F(parent_1 | parent_2)
            q  ~ Uniform(0, 1)
            wy = F(child | parent_2) sampled from C13|2 conditional on w1
            child = inverse F(child | parent_2)
        """
        record = self._get_conditional_record(
            child_name=child_name,
            parent_names=parent_names,
        )

        record_parent_1 = record["parent_1"]
        record_parent_2 = record["parent_2"]
        record_child = record["child"]


        self._selected_order_from_record(record)

        if record_child != child_name:
            raise RuntimeError(
                "Conditional record has unexpected child.\n"
                f"Expected: {child_name}\n"
                f"Got: {record_child}"
            )

        c12 = self._load_bicop(record["bicop_parent_1_parent_2_file"])
        c23 = self._load_bicop(record["bicop_parent_2_child_file"])

        conditional_model_type = record.get(
            "conditional_model_type",
            "simplified",
        )
        selected_number_bins = int(record.get("selected_number_bins", 1))

        p1_idx = self.column_to_index[record_parent_1]
        p2_idx = self.column_to_index[record_parent_2]

        u1 = np.asarray(u_matrix[:, p1_idx], dtype=float)
        u2 = np.asarray(u_matrix[:, p2_idx], dtype=float)

        u1 = np.clip(u1, self.clip_eps, 1.0 - self.clip_eps)
        u2 = np.clip(u2, self.clip_eps, 1.0 - self.clip_eps)

        n_samples = len(u1)

        # w1 = F(parent_1 | parent_2)
        data12 = np.asfortranarray(np.column_stack([u1, u2]))
        w1 = c12.hfunc2(data12)
        w1 = np.clip(w1, self.clip_eps, 1.0 - self.clip_eps)

        # Sample wy = F(child | parent_2) conditional on w1 using C13|2.
        q = rng.uniform(
            self.clip_eps,
            1.0 - self.clip_eps,
            size=n_samples,
        )

        if conditional_model_type == "quantile_binned" and selected_number_bins > 1:
            bin_edges = np.asarray(
                record.get("conditional_bin_edges", []),
                dtype=float,
            ).reshape(-1)
            bin_model_files = list(
                record.get("conditional_bin_model_files", [])
            )
            if len(bin_edges) != selected_number_bins + 1:
                raise ValueError(
                    "Binned conditional record has inconsistent bin edges: "
                    f"expected {selected_number_bins + 1}, got {len(bin_edges)}."
                )
            if len(bin_model_files) != selected_number_bins:
                raise ValueError(
                    "Binned conditional record has inconsistent model files: "
                    f"expected {selected_number_bins}, got {len(bin_model_files)}."
                )
            if np.any(np.diff(bin_edges) <= 0.0):
                raise ValueError("Conditional bin edges must be strictly increasing.")

            bin_indices = np.clip(
                np.searchsorted(bin_edges[1:-1], u2, side="right"),
                0,
                selected_number_bins - 1,
            ).astype(int)
            wy = np.empty(n_samples, dtype=float)
            for bin_index, model_file in enumerate(bin_model_files):
                mask = bin_indices == bin_index
                if not np.any(mask):
                    continue
                conditional_model = self._load_bicop(model_file)
                inverse_data = np.asfortranarray(
                    np.column_stack([w1[mask], q[mask]])
                )
                wy[mask] = conditional_model.hinv1(inverse_data)
        else:
            c13_given_2 = self._load_bicop(
                record["bicop_parent_1_child_given_parent_2_file"]
            )
            data13_inverse = np.asfortranarray(np.column_stack([w1, q]))

            # C13|2 is fitted on [w1, wy].
            # hinv1 inverts h1(w1, wy) w.r.t. wy.
            wy = c13_given_2.hinv1(data13_inverse)

        wy = np.clip(wy, self.clip_eps, 1.0 - self.clip_eps)

        # Invert wy = F(child | parent_2) using C23 fitted on [parent_2, child].
        data23_inverse = np.asfortranarray(np.column_stack([u2, wy]))

        # hinv1 inverts h1(parent_2, child) w.r.t. child.
        u_child = c23.hinv1(data23_inverse)
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