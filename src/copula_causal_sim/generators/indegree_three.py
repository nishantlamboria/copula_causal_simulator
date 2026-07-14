"""Graph-conditioned generation for DAGs with maximum indegree three."""

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
        "pyvinecopulib is required for indegree-three generation."
    ) from exc

from copula_causal_sim.copulas.marginals import EmpiricalMarginalLibrary
from copula_causal_sim.generators.indegree_two import IndegreeTwoGenerator
from copula_causal_sim.graphs.dag import parents_of, topological_order, validate_dag


@dataclass
class IndegreeThreeGenerationResult:
    u: pd.DataFrame
    x: pd.DataFrame
    adjacency: np.ndarray
    columns: list[str]
    seed: int
    max_supported_indegree: int = 3


def load_conditional_vine_metadata_indegree3(
    path: str | Path,
) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Indegree-three conditional vine metadata not found: {path}"
        )
    with path.open("rb") as file:
        payload = pickle.load(file)
    required = ["dataset_id", "columns", "parent_set_size", "records"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Invalid indegree-three metadata. Missing keys: {missing}")
    if int(payload["parent_set_size"]) != 3:
        raise ValueError(
            "This generator requires a conditional vine library with "
            "parent_set_size=3."
        )
    return payload


class IndegreeThreeGenerator:
    """Generator for roots and local mechanisms with one, two, or three parents."""

    def __init__(
        self,
        marginal_library: EmpiricalMarginalLibrary,
        paircopula_metadata: dict[str, Any],
        conditional_vine_metadata_indegree2: dict[str, Any],
        conditional_vine_metadata_indegree3: dict[str, Any],
        artifact_dir: str | Path,
        clip_eps: float = 1e-6,
    ) -> None:
        self.marginal_library = marginal_library
        self.paircopula_metadata = paircopula_metadata
        self.conditional_vine_metadata_indegree2 = conditional_vine_metadata_indegree2
        self.conditional_vine_metadata_indegree3 = conditional_vine_metadata_indegree3
        self.artifact_dir = Path(artifact_dir)
        self.clip_eps = float(clip_eps)
        self.columns = list(marginal_library.columns)
        self.column_to_index = {
            column: index for index, column in enumerate(self.columns)
        }

        for label, metadata in [
            ("pair-copula", paircopula_metadata),
            ("indegree-two", conditional_vine_metadata_indegree2),
            ("indegree-three", conditional_vine_metadata_indegree3),
        ]:
            metadata_columns = list(metadata["columns"])
            if metadata_columns != self.columns:
                raise ValueError(
                    f"Column mismatch between marginals and {label} metadata.\n"
                    f"Marginals: {self.columns}\nMetadata: {metadata_columns}"
                )

        self.lower_order_generator = IndegreeTwoGenerator(
            marginal_library=marginal_library,
            paircopula_metadata=paircopula_metadata,
            conditional_vine_metadata=conditional_vine_metadata_indegree2,
            artifact_dir=artifact_dir,
            clip_eps=clip_eps,
        )
        self.records = conditional_vine_metadata_indegree3["records"]
        self.lookup = self._build_lookup(self.records)
        self.bicop_cache: dict[str, Any] = {}

    @staticmethod
    def _build_lookup(
        records: list[dict[str, Any]],
    ) -> dict[tuple[str, frozenset[str]], dict[str, Any]]:
        lookup: dict[tuple[str, frozenset[str]], dict[str, Any]] = {}
        for record in records:
            if record.get("status") != "ok":
                continue
            child = str(record["child"])
            parent_set = frozenset(
                record.get(
                    "parent_set",
                    [record["parent_1"], record["parent_2"], record["parent_3"]],
                )
            )
            if len(parent_set) != 3:
                raise ValueError(
                    f"Invalid three-parent record for child={child!r}: "
                    f"{sorted(parent_set)!r}."
                )
            key = (child, parent_set)
            if key in lookup:
                raise ValueError(
                    "Duplicate indegree-three records for "
                    f"child={child!r}, parents={sorted(parent_set)!r}."
                )
            lookup[key] = record
        return lookup

    @staticmethod
    def _selected_order(record: dict[str, Any]) -> list[str]:
        expected = [
            record["parent_1"],
            record["parent_2"],
            record["parent_3"],
            record["child"],
        ]
        selected = record.get("selected_order", expected)
        if not isinstance(selected, (list, tuple)) or list(selected) != expected:
            raise ValueError(
                "Indegree-three record has inconsistent selected_order. "
                f"Expected {expected!r}, got {selected!r}."
            )
        return list(selected)

    def _get_record(
        self,
        child_name: str,
        parent_names: list[str],
    ) -> dict[str, Any]:
        if len(parent_names) != 3 or len(set(parent_names)) != 3:
            raise ValueError("Exactly three distinct parent names are required.")
        key = (child_name, frozenset(parent_names))
        if key not in self.lookup:
            raise KeyError(
                "No fitted four-dimensional D-vine found for local mechanism:\n"
                f"child={child_name}, parents={parent_names}"
            )
        return self.lookup[key]

    def _resolve_model_path(self, stored_path: str | Path) -> Path:
        stored = Path(stored_path)
        if stored.exists():
            return stored
        fallback = (
            self.artifact_dir / "conditional_vines_indegree3" / stored.name
        )
        if fallback.exists():
            return fallback
        raise FileNotFoundError(
            "Could not find fitted indegree-three Bicop model.\n"
            f"Stored path: {stored}\nFallback path: {fallback}"
        )

    def _load_bicop(self, stored_path: str | Path):
        model_path = self._resolve_model_path(stored_path)
        key = str(model_path)
        if key not in self.bicop_cache:
            self.bicop_cache[key] = pv.Bicop.from_file(str(model_path))
        return self.bicop_cache[key]

    def _sample_child_given_three_parents(
        self,
        child_name: str,
        parent_names: list[str],
        u_matrix: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        record = self._get_record(child_name, parent_names)
        self._selected_order(record)

        p1 = str(record["parent_1"])
        p2 = str(record["parent_2"])
        p3 = str(record["parent_3"])
        if str(record["child"]) != child_name:
            raise RuntimeError("Stored indegree-three record has the wrong child.")

        c12 = self._load_bicop(record["bicop_parent_1_parent_2_file"])
        c23 = self._load_bicop(record["bicop_parent_2_parent_3_file"])
        c3x = self._load_bicop(record["bicop_parent_3_child_file"])
        c13_2 = self._load_bicop(
            record["bicop_parent_1_parent_3_given_parent_2_file"]
        )
        c2x_3 = self._load_bicop(
            record["bicop_parent_2_child_given_parent_3_file"]
        )
        c1x_23 = self._load_bicop(
            record["bicop_parent_1_child_given_parent_2_parent_3_file"]
        )

        u1 = np.clip(
            u_matrix[:, self.column_to_index[p1]],
            self.clip_eps,
            1.0 - self.clip_eps,
        )
        u2 = np.clip(
            u_matrix[:, self.column_to_index[p2]],
            self.clip_eps,
            1.0 - self.clip_eps,
        )
        u3 = np.clip(
            u_matrix[:, self.column_to_index[p3]],
            self.clip_eps,
            1.0 - self.clip_eps,
        )

        data12 = np.asfortranarray(np.column_stack([u1, u2]), dtype=float)
        data23 = np.asfortranarray(np.column_stack([u2, u3]), dtype=float)

        w1_given_2 = np.clip(
            c12.hfunc2(data12), self.clip_eps, 1.0 - self.clip_eps
        )
        w3_given_2 = np.clip(
            c23.hfunc1(data23), self.clip_eps, 1.0 - self.clip_eps
        )
        w1_given_23 = np.clip(
            c13_2.hfunc2(
                np.asfortranarray(
                    np.column_stack([w1_given_2, w3_given_2]),
                    dtype=float,
                )
            ),
            self.clip_eps,
            1.0 - self.clip_eps,
        )
        w2_given_3 = np.clip(
            c23.hfunc2(data23), self.clip_eps, 1.0 - self.clip_eps
        )

        q = rng.uniform(
            self.clip_eps,
            1.0 - self.clip_eps,
            size=len(u1),
        )
        wx_given_23 = np.clip(
            c1x_23.hinv1(
                np.asfortranarray(
                    np.column_stack([w1_given_23, q]), dtype=float
                )
            ),
            self.clip_eps,
            1.0 - self.clip_eps,
        )
        wx_given_3 = np.clip(
            c2x_3.hinv1(
                np.asfortranarray(
                    np.column_stack([w2_given_3, wx_given_23]), dtype=float
                )
            ),
            self.clip_eps,
            1.0 - self.clip_eps,
        )
        ux = np.asarray(
            c3x.hinv1(
                np.asfortranarray(
                    np.column_stack([u3, wx_given_3]), dtype=float
                )
            ),
            dtype=float,
        )
        ux = np.clip(ux, self.clip_eps, 1.0 - self.clip_eps)
        if not np.isfinite(ux).all():
            raise RuntimeError("Generated three-parent child values are non-finite.")
        return ux

    def sample_u(
        self,
        adjacency: np.ndarray,
        n_samples: int,
        seed: int,
    ) -> pd.DataFrame:
        if int(n_samples) <= 0:
            raise ValueError("n_samples must be positive.")
        adj = validate_dag(adjacency, max_allowed_indegree=3)
        if adj.shape[0] != len(self.columns):
            raise ValueError(
                "Graph size does not match the number of dataset variables.\n"
                f"Graph nodes: {adj.shape[0]}\nVariables: {len(self.columns)}"
            )

        rng = np.random.default_rng(int(seed))
        u = np.full((n_samples, len(self.columns)), np.nan, dtype=float)

        for node in topological_order(adj):
            node_name = self.columns[node]
            parents = parents_of(adj, node)
            parent_names = [self.columns[index] for index in parents]

            if len(parents) == 0:
                u[:, node] = rng.uniform(
                    self.clip_eps, 1.0 - self.clip_eps, size=n_samples
                )
            elif len(parents) == 1:
                parent = parents[0]
                u[:, node] = (
                    self.lower_order_generator.one_parent_generator
                    ._sample_child_given_parent(
                        parent_name=self.columns[parent],
                        child_name=node_name,
                        u_parent=u[:, parent],
                        rng=rng,
                    )
                )
            elif len(parents) == 2:
                u[:, node] = self.lower_order_generator._sample_child_given_two_parents(
                    child_name=node_name,
                    parent_names=parent_names,
                    u_matrix=u,
                    rng=rng,
                )
            elif len(parents) == 3:
                u[:, node] = self._sample_child_given_three_parents(
                    child_name=node_name,
                    parent_names=parent_names,
                    u_matrix=u,
                    rng=rng,
                )
            else:
                raise ValueError(
                    f"Node {node_name} has {len(parents)} parents. "
                    "This generator supports maximum indegree three."
                )

        if np.isnan(u).any() or not np.isfinite(u).all():
            raise RuntimeError("Generated copula-space data contain invalid values.")
        u = np.clip(u, self.clip_eps, 1.0 - self.clip_eps)
        return pd.DataFrame(u, columns=self.columns)

    def generate(
        self,
        adjacency: np.ndarray,
        n_samples: int,
        seed: int,
    ) -> IndegreeThreeGenerationResult:
        u_df = self.sample_u(adjacency, n_samples, seed)
        x_df = self.marginal_library.inverse_transform(u_df)
        return IndegreeThreeGenerationResult(
            u=u_df,
            x=x_df,
            adjacency=np.asarray(adjacency, dtype=int),
            columns=self.columns,
            seed=int(seed),
            max_supported_indegree=3,
        )
