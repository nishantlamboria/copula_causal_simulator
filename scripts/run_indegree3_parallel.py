"""Parallel exhaustive indegree-three copula fitting for Windows/local machines.

Fits independent child/three-parent mechanisms in separate processes while
keeping pyvinecopulib at one internal thread per worker. The run is resumable.
"""

from __future__ import annotations

import os

# Prevent each worker from creating additional BLAS/OpenMP thread pools.
for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import argparse
import pickle
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from copula_causal_sim.config import load_config
from copula_causal_sim.data.loaders import load_tabular_dataset
from copula_causal_sim.copulas.marginals import make_pseudo_observations
from copula_causal_sim.copulas.conditional_vine_library import (
    build_bicop_fit_controls,
)
from copula_causal_sim.copulas.conditional_vine_indegree3 import (
    ConditionalVineIndegree3Library,
    ConditionalVineIndegree3Record,
    fit_one_conditional_vine_indegree3,
)


_WORKER_U_DF: pd.DataFrame | None = None
_WORKER_CONTROLS: Any = None
_WORKER_FAMILY_NAMES: list[str] | None = None
_WORKER_SETTINGS: dict[str, Any] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit an exhaustive indegree-three conditional-vine library "
            "using multiple independent Python processes."
        )
    )
    parser.add_argument("--config", required=True, help="Dataset YAML config.")
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of Python worker processes. Use 8 initially.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of mechanisms for benchmarking.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Save resumable progress after this many completed mechanisms.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Delete prior indegree-three output for this config and start again.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Refit records whose previous status was 'failed'.",
    )
    return parser.parse_args()


def task_key(child: str, parents: tuple[str, str, str] | list[str]) -> tuple[str, tuple[str, str, str]]:
    canonical = tuple(sorted(str(value) for value in parents))
    if len(canonical) != 3:
        raise ValueError("Expected exactly three parents.")
    return str(child), canonical  # type: ignore[return-value]


def save_checkpoint(
    path: Path,
    *,
    dataset_id: str,
    records_by_key: dict[tuple[str, tuple[str, str, str]], dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_id": dataset_id,
        "records_by_key": records_by_key,
    }
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def load_checkpoint(
    path: Path,
    *,
    dataset_id: str,
) -> dict[tuple[str, tuple[str, str, str]], dict[str, Any]]:
    if not path.exists():
        return {}

    with path.open("rb") as file:
        payload = pickle.load(file)

    if payload.get("dataset_id") != dataset_id:
        raise RuntimeError(
            f"Checkpoint dataset mismatch: expected {dataset_id!r}, "
            f"found {payload.get('dataset_id')!r}."
        )

    return dict(payload.get("records_by_key", {}))


def worker_initialise(
    pseudo_path: str,
    settings: dict[str, Any],
) -> None:
    global _WORKER_U_DF
    global _WORKER_CONTROLS
    global _WORKER_FAMILY_NAMES
    global _WORKER_SETTINGS

    _WORKER_U_DF = pd.read_csv(pseudo_path)
    _WORKER_SETTINGS = settings
    _WORKER_CONTROLS, _WORKER_FAMILY_NAMES = build_bicop_fit_controls(
        selection_criterion=settings["selection_criterion"],
        allow_rotations=settings["allow_rotations"],
        num_threads=1,
    )


def worker_fit(
    task: tuple[str, tuple[str, str, str]],
) -> dict[str, Any]:
    if (
        _WORKER_U_DF is None
        or _WORKER_CONTROLS is None
        or _WORKER_FAMILY_NAMES is None
        or _WORKER_SETTINGS is None
    ):
        raise RuntimeError("Worker was not initialised.")

    child, parents = task
    settings = _WORKER_SETTINGS

    record = fit_one_conditional_vine_indegree3(
        u_df=_WORKER_U_DF,
        child=child,
        parents=parents,
        dataset_id=settings["dataset_id"],
        controls=_WORKER_CONTROLS,
        candidate_families=_WORKER_FAMILY_NAMES,
        selection_criterion=settings["selection_criterion"],
        model_dir=settings["model_dir"],
        clip_eps=settings["clip_eps"],
        order_strategy=settings["order_strategy"],
        order_validation_fraction=settings["order_validation_fraction"],
        order_selection_seed=settings["order_selection_seed"],
        minimum_validation_rows=settings["minimum_validation_rows"],
    )

    return asdict(record)


def main() -> None:
    args = parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be positive.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive.")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be positive.")

    config_path = (PROJECT_ROOT / args.config).resolve()
    config = load_config(config_path)

    dataset_id = str(config["dataset_id"])
    data_config = config.get("data", {})
    copula_config = config.get("copula", {})

    clip_eps = float(data_config.get("clip_eps", 1e-6))
    selection_criterion = str(
        copula_config.get("selection_criterion", "bic")
    )
    allow_rotations = bool(
        copula_config.get("allow_rotations", True)
    )
    order_strategy = str(
        copula_config.get(
            "vine_order_strategy",
            "heldout_conditional_loglik",
        )
    )
    order_validation_fraction = float(
        copula_config.get(
            "vine_order_validation_fraction",
            0.20,
        )
    )
    order_selection_seed = int(
        copula_config.get(
            "vine_order_selection_seed",
            42,
        )
    )
    minimum_validation_rows = int(
        copula_config.get(
            "vine_order_minimum_validation_rows",
            50,
        )
    )

    artifact_dir = (PROJECT_ROOT / config["outputs"]["artifact_dir"]).resolve()
    table_dir = (PROJECT_ROOT / config["outputs"]["table_dir"]).resolve()
    model_dir = artifact_dir / "conditional_vines_indegree3"
    pseudo_path = artifact_dir / "pseudo_observations.csv"
    checkpoint_path = (
        artifact_dir / "conditional_vine_library_indegree3.checkpoint.pkl"
    )
    final_library_path = (
        artifact_dir / "conditional_vine_library_indegree3.pkl"
    )
    final_summary_path = (
        table_dir / "conditional_vine_summary_indegree3.csv"
    )

    if args.restart:
        checkpoint_path.unlink(missing_ok=True)
        final_library_path.unlink(missing_ok=True)
        final_summary_path.unlink(missing_ok=True)
        if model_dir.exists():
            shutil.rmtree(model_dir)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    if pseudo_path.exists():
        u_df = pd.read_csv(pseudo_path)
        print(f"Loading pseudo-observations: {pseudo_path}")
    else:
        print("Creating pseudo-observations...")
        df = load_tabular_dataset(config)
        u_df = make_pseudo_observations(df, clip_eps=clip_eps)
        u_df.to_csv(pseudo_path, index=False)
        print(f"Saved pseudo-observations: {pseudo_path}")

    columns = list(u_df.columns)
    tasks: list[tuple[str, tuple[str, str, str]]] = []

    for child in columns:
        possible_parents = [
            column for column in columns if column != child
        ]
        for parents in combinations(possible_parents, 3):
            tasks.append((child, parents))

    if args.limit is not None:
        tasks = tasks[: args.limit]

    records_by_key = load_checkpoint(
        checkpoint_path,
        dataset_id=dataset_id,
    )

    if args.retry_failed:
        records_by_key = {
            key: record
            for key, record in records_by_key.items()
            if record.get("status") != "failed"
        }

    task_keys = {
        task_key(child, parents)
        for child, parents in tasks
    }
    records_by_key = {
        key: record
        for key, record in records_by_key.items()
        if key in task_keys
    }

    remaining_tasks = [
        task
        for task in tasks
        if task_key(*task) not in records_by_key
    ]

    settings = {
        "dataset_id": dataset_id,
        "selection_criterion": selection_criterion,
        "allow_rotations": allow_rotations,
        "model_dir": str(model_dir),
        "clip_eps": clip_eps,
        "order_strategy": order_strategy,
        "order_validation_fraction": order_validation_fraction,
        "order_selection_seed": order_selection_seed,
        "minimum_validation_rows": minimum_validation_rows,
    }

    _, family_names = build_bicop_fit_controls(
        selection_criterion=selection_criterion,
        allow_rotations=allow_rotations,
        num_threads=1,
    )

    print("=" * 72)
    print(f"Dataset:              {dataset_id}")
    print(f"Rows x columns:       {u_df.shape}")
    print(f"Total mechanisms:     {len(tasks)}")
    print(f"Already completed:    {len(records_by_key)}")
    print(f"Remaining mechanisms: {len(remaining_tasks)}")
    print(f"Worker processes:     {args.workers}")
    print("Threads per worker:   1")
    print(f"Model directory:      {model_dir}")
    print("=" * 72)

    started = time.perf_counter()
    completed_this_run = 0

    if remaining_tasks:
        executor = ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=worker_initialise,
            initargs=(str(pseudo_path), settings),
        )

        futures = {
            executor.submit(worker_fit, task): task
            for task in remaining_tasks
        }

        try:
            for future in as_completed(futures):
                child, parents = futures[future]
                record = future.result()
                key = task_key(child, parents)
                records_by_key[key] = record
                completed_this_run += 1

                elapsed = time.perf_counter() - started
                done = len(records_by_key)
                rate = completed_this_run / elapsed if elapsed > 0 else 0.0
                remaining = len(tasks) - done
                eta_seconds = remaining / rate if rate > 0 else float("inf")
                eta_minutes = (
                    eta_seconds / 60.0
                    if eta_seconds != float("inf")
                    else float("inf")
                )

                print(
                    f"[{done:04d}/{len(tasks):04d}] "
                    f"{record['status'].upper():6s} "
                    f"{child} | {', '.join(parents)} | "
                    f"elapsed={elapsed / 60.0:.1f} min | "
                    f"ETA={eta_minutes:.1f} min",
                    flush=True,
                )

                if completed_this_run % args.checkpoint_every == 0:
                    save_checkpoint(
                        checkpoint_path,
                        dataset_id=dataset_id,
                        records_by_key=records_by_key,
                    )

        except BaseException:
            save_checkpoint(
                checkpoint_path,
                dataset_id=dataset_id,
                records_by_key=records_by_key,
            )
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)

    save_checkpoint(
        checkpoint_path,
        dataset_id=dataset_id,
        records_by_key=records_by_key,
    )

    missing_keys = [
        task_key(*task)
        for task in tasks
        if task_key(*task) not in records_by_key
    ]
    if missing_keys:
        raise RuntimeError(
            f"Run ended with {len(missing_keys)} missing mechanisms."
        )

    ordered_records = [
        ConditionalVineIndegree3Record(
            **records_by_key[task_key(*task)]
        )
        for task in tasks
    ]

    library = ConditionalVineIndegree3Library(
        dataset_id=dataset_id,
        columns=columns,
        n_rows=len(u_df),
        parent_set_size=3,
        selection_criterion=selection_criterion,
        family_set=family_names,
        order_strategy=order_strategy,
        order_validation_fraction=order_validation_fraction,
        order_selection_seed=order_selection_seed,
        minimum_validation_rows=minimum_validation_rows,
        records=ordered_records,
    )

    summary = library.to_dataframe()
    summary.to_csv(final_summary_path, index=False)
    library.save_pickle(final_library_path)

    n_ok = int((summary["status"] == "ok").sum())
    n_failed = int((summary["status"] == "failed").sum())
    elapsed = time.perf_counter() - started

    print("=" * 72)
    print("Parallel indegree-three build complete.")
    print(f"Dataset:         {dataset_id}")
    print(f"Attempted:       {len(summary)}")
    print(f"Successful:      {n_ok}")
    print(f"Failed:          {n_failed}")
    print(f"Elapsed:         {elapsed / 60.0:.2f} minutes")
    print(f"Summary:         {final_summary_path}")
    print(f"Library:         {final_library_path}")
    print(f"Models:          {model_dir}")
    print(f"Checkpoint:      {checkpoint_path}")
    print("=" * 72)

    if n_failed:
        print(
            summary.loc[
                summary["status"] == "failed",
                ["child", "parent_set", "error"],
            ].head(20).to_string(index=False)
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
