from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from copula_causal_sim.config import load_config
from copula_causal_sim.data.loaders import load_tabular_dataset


ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "sachs": 1320,
    "diabetes": 840,
    "breast_cancer": 1980,
}

EXPECTED_SAMPLE_SIZES = {500, 1000, 2000}
EXPECTED_SEEDS = {1, 2, 3, 4, 5}


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def parse_boolean_column(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series

    values = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }

    unknown = sorted(set(values.dropna()) - set(mapping))
    if unknown:
        raise ValueError(f"Unsupported boolean values: {unknown}")

    return values.map(mapping)


def is_dag(adjacency: np.ndarray) -> bool:
    indegrees = adjacency.sum(axis=0).astype(int)
    queue = [
        index
        for index, degree in enumerate(indegrees)
        if degree == 0
    ]
    visited = 0

    while queue:
        node = queue.pop()
        visited += 1

        for child in np.flatnonzero(adjacency[node]):
            indegrees[child] -= 1
            if indegrees[child] == 0:
                queue.append(int(child))

    return visited == adjacency.shape[0]


def check_library(dataset: str, expected_mechanisms: int) -> None:
    config_path = ROOT / "configs" / "datasets" / f"{dataset}.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    artifact_dir = resolve_path(str(config["outputs"]["artifact_dir"]))
    table_dir = resolve_path(str(config["outputs"]["table_dir"]))

    library_path = artifact_dir / "conditional_vine_library_indegree3.pkl"
    model_dir = artifact_dir / "conditional_vines_indegree3"
    summary_path = table_dir / "conditional_vine_summary_indegree3.csv"

    for path in (library_path, model_dir, summary_path):
        if not path.exists():
            raise FileNotFoundError(path)

    summary = pd.read_csv(summary_path)

    if "status" not in summary.columns:
        raise KeyError(f"'status' column missing from {summary_path}")

    statuses = summary["status"].astype(str).str.strip().str.lower()
    successful = int((statuses == "ok").sum())
    failed = int((statuses == "failed").sum())

    if len(summary) != expected_mechanisms:
        raise RuntimeError(
            f"{dataset}: expected {expected_mechanisms} rows, "
            f"found {len(summary)}."
        )

    if successful != expected_mechanisms:
        raise RuntimeError(
            f"{dataset}: expected {expected_mechanisms} successful "
            f"mechanisms, found {successful}."
        )

    if failed != 0:
        raise RuntimeError(f"{dataset}: found {failed} failed mechanisms.")

    model_count = len(list(model_dir.rglob("*.json")))
    if model_count == 0:
        raise RuntimeError(f"{dataset}: no JSON model files found.")

    print(
        f"PASS  {dataset:15s} library: "
        f"{successful}/{expected_mechanisms} successful, "
        f"{model_count} JSON files"
    )


def check_end_to_end(dataset: str) -> None:
    config_path = ROOT / "configs" / "datasets" / f"{dataset}.yaml"
    config = load_config(config_path)
    original = load_tabular_dataset(config)
    expected_columns = list(original.columns)

    graph_path = (
        ROOT
        / "graphs"
        / "end_to_end"
        / f"{dataset}_mixed_indegree.csv"
    )
    output_dir = ROOT / "outputs" / "end_to_end" / f"{dataset}_seed42"

    u_path = output_dir / "generated_u.csv"
    x_path = output_dir / "generated_x.csv"
    metadata_path = output_dir / "generation_metadata.json"

    for path in (graph_path, u_path, x_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    generated_u = pd.read_csv(u_path)
    generated_x = pd.read_csv(x_path)
    adjacency = pd.read_csv(graph_path, header=None).to_numpy(dtype=int)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    expected_shape = (2000, len(expected_columns))

    if generated_u.shape != expected_shape:
        raise RuntimeError(
            f"{dataset}: generated_u shape is {generated_u.shape}, "
            f"expected {expected_shape}."
        )

    if generated_x.shape != expected_shape:
        raise RuntimeError(
            f"{dataset}: generated_x shape is {generated_x.shape}, "
            f"expected {expected_shape}."
        )

    if list(generated_u.columns) != expected_columns:
        raise RuntimeError(f"{dataset}: generated_u columns do not match.")

    if list(generated_x.columns) != expected_columns:
        raise RuntimeError(f"{dataset}: generated_x columns do not match.")

    u_values = generated_u.to_numpy(dtype=float)
    x_values = generated_x.to_numpy(dtype=float)

    if not np.isfinite(u_values).all():
        raise RuntimeError(f"{dataset}: generated_u contains non-finite values.")

    if not np.isfinite(x_values).all():
        raise RuntimeError(f"{dataset}: generated_x contains non-finite values.")

    if u_values.min() < 0.0 or u_values.max() > 1.0:
        raise RuntimeError(f"{dataset}: generated_u lies outside [0, 1].")

    expected_graph_shape = (len(expected_columns), len(expected_columns))

    if adjacency.shape != expected_graph_shape:
        raise RuntimeError(
            f"{dataset}: adjacency shape is {adjacency.shape}, "
            f"expected {expected_graph_shape}."
        )

    if np.any(np.diag(adjacency) != 0):
        raise RuntimeError(f"{dataset}: graph contains self-loops.")

    if not is_dag(adjacency):
        raise RuntimeError(f"{dataset}: graph is not acyclic.")

    maximum_indegree = int(adjacency.sum(axis=0).max())
    if maximum_indegree != 3:
        raise RuntimeError(
            f"{dataset}: expected maximum indegree 3, "
            f"found {maximum_indegree}."
        )

    metadata_dataset = metadata.get("dataset_id", metadata.get("dataset"))

    if metadata_dataset != dataset:
        raise RuntimeError(
            f"{dataset}: metadata dataset is {metadata_dataset!r}."
        )

    if int(metadata.get("n_samples", -1)) != 2000:
        raise RuntimeError(f"{dataset}: metadata n_samples is incorrect.")

    if int(metadata.get("seed", -1)) != 42:
        raise RuntimeError(f"{dataset}: metadata seed is incorrect.")

    print(
        f"PASS  {dataset:15s} end-to-end generation: "
        "2000 rows, max indegree=3"
    )


def check_multiseed_validation() -> None:
    validation_root = (
        ROOT
        / "reports"
        / "final_validation"
        / "indegree3_multiseed"
    )

    all_runs_path = validation_root / "all_runs.csv"
    summary_path = validation_root / "summary_by_sample_size.csv"
    overall_path = validation_root / "overall_summary.csv"

    for path in (all_runs_path, summary_path, overall_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    runs = pd.read_csv(all_runs_path)

    if len(runs) != 15:
        raise RuntimeError(
            f"Expected 15 validation runs, found {len(runs)}."
        )

    expected_combinations = {
        (sample_size, seed)
        for sample_size in EXPECTED_SAMPLE_SIZES
        for seed in EXPECTED_SEEDS
    }

    actual_combinations = {
        (int(row.n), int(row.seed))
        for row in runs.itertuples()
    }

    if actual_combinations != expected_combinations:
        missing = expected_combinations - actual_combinations
        extra = actual_combinations - expected_combinations
        raise RuntimeError(
            "Validation combinations are incorrect. "
            f"Missing={sorted(missing)}, extra={sorted(extra)}"
        )

    finite = parse_boolean_column(runs["generated_values_finite"])
    bounded = parse_boolean_column(
        runs["generated_values_in_unit_interval"]
    )
    all_orders = parse_boolean_column(runs["all_six_orders_scored"])

    if not finite.all():
        raise RuntimeError("At least one run produced non-finite values.")

    if not bounded.all():
        raise RuntimeError(
            "At least one run produced values outside [0, 1]."
        )

    if not all_orders.all():
        raise RuntimeError(
            "At least one run did not score all six candidate orders."
        )

    print(
        "PASS  multi-seed validation: "
        "15/15 runs present and valid"
    )


def main() -> None:
    print("=" * 78)
    print("FINAL INDEGREE-THREE RELEASE CHECK")
    print("=" * 78)

    for dataset, expected in DATASETS.items():
        check_library(dataset, expected)

    print()

    for dataset in DATASETS:
        check_end_to_end(dataset)

    print()

    check_multiseed_validation()

    print()
    print("=" * 78)
    print("All final indegree-three technical checks passed.")
    print("=" * 78)


if __name__ == "__main__":
    main()
