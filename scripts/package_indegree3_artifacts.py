from __future__ import annotations

import hashlib
import json
import os
import tarfile
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path.cwd()

DATASETS = {
    "sachs": 1320,
    "diabetes": 840,
    "breast_cancer": 1980,
}

OUTPUT_DIR = ROOT / "artifacts" / "precomputed_indegree3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def resolve_project_path(value: str) -> Path:
    """Resolve a config path relative to the repository root."""
    path = Path(value)

    if path.is_absolute():
        return path

    return ROOT / path


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def package_dataset(dataset: str, expected: int) -> tuple[str, dict]:
    config_path = (
        ROOT
        / "configs"
        / "datasets"
        / f"{dataset}.yaml"
    )

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    config = yaml.safe_load(
        config_path.read_text(encoding="utf-8")
    )

    outputs = config.get("outputs", {})

    if "artifact_dir" not in outputs:
        raise KeyError(
            f"'outputs.artifact_dir' is missing from {config_path}"
        )

    if "table_dir" not in outputs:
        raise KeyError(
            f"'outputs.table_dir' is missing from {config_path}"
        )

    artifact_dir = resolve_project_path(
        outputs["artifact_dir"]
    )
    table_dir = resolve_project_path(
        outputs["table_dir"]
    )

    library_path = (
        artifact_dir
        / "conditional_vine_library_indegree3.pkl"
    )
    model_dir = (
        artifact_dir
        / "conditional_vines_indegree3"
    )
    summary_path = (
        table_dir
        / "conditional_vine_summary_indegree3.csv"
    )
    pseudo_path = (
        artifact_dir
        / "pseudo_observations.csv"
    )

    required_paths = [
        library_path,
        model_dir,
        summary_path,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required output not found for {dataset}: {path}"
            )

    summary = pd.read_csv(summary_path)

    if "status" not in summary.columns:
        raise KeyError(
            f"'status' column not found in {summary_path}"
        )

    status = (
        summary["status"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    successful = int((status == "ok").sum())
    failed = int((status == "failed").sum())
    total_rows = len(summary)

    if total_rows != expected:
        raise RuntimeError(
            f"{dataset}: expected {expected} rows, "
            f"but found {total_rows}"
        )

    if successful != expected or failed != 0:
        raise RuntimeError(
            f"{dataset}: validation failed: "
            f"rows={total_rows}, ok={successful}, failed={failed}"
        )

    model_files = list(model_dir.rglob("*.json"))

    archive_path = (
        OUTPUT_DIR
        / f"{dataset}_indegree3.tar.gz"
    )
    temporary_archive = archive_path.with_name(
        archive_path.name + ".tmp"
    )

    temporary_archive.unlink(missing_ok=True)

    print()
    print("=" * 72)
    print(f"Dataset:       {dataset}")
    print(f"Mechanisms:    {total_rows}")
    print(f"Model files:   {len(model_files)}")
    print(f"Artifact dir:  {artifact_dir}")
    print(f"Table dir:     {table_dir}")
    print(f"Creating:      {archive_path}")
    print("=" * 72)

    with tarfile.open(
        temporary_archive,
        mode="w:gz",
        compresslevel=6,
    ) as archive:
        archive.add(
            library_path,
            arcname=library_path.name,
        )

        archive.add(
            model_dir,
            arcname=model_dir.name,
        )

        archive.add(
            summary_path,
            arcname=summary_path.name,
        )

        if pseudo_path.is_file():
            archive.add(
                pseudo_path,
                arcname=pseudo_path.name,
            )

    os.replace(
        temporary_archive,
        archive_path,
    )

    archive_checksum = calculate_sha256(
        archive_path
    )
    library_checksum = calculate_sha256(
        library_path
    )

    extract_into = Path(
        outputs["artifact_dir"]
    ).as_posix()

    manifest = {
        "dataset": dataset,
        "expected_mechanisms": expected,
        "summary_rows": total_rows,
        "successful_mechanisms": successful,
        "failed_mechanisms": failed,
        "model_json_files": len(model_files),
        "archive": archive_path.name,
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_sha256": archive_checksum,
        "library_sha256": library_checksum,
        "extract_into": extract_into,
        "included_files": [
            library_path.name,
            model_dir.name,
            summary_path.name,
        ]
        + (
            [pseudo_path.name]
            if pseudo_path.is_file()
            else []
        ),
    }

    manifest_path = (
        OUTPUT_DIR
        / f"{dataset}_indegree3_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    size_mb = archive_path.stat().st_size / (1024 ** 2)

    print(
        f"Finished {dataset}: "
        f"{size_mb:.2f} MB"
    )
    print(
        f"SHA-256: {archive_checksum}"
    )

    checksum_line = (
        f"{archive_checksum}  {archive_path.name}"
    )

    return checksum_line, manifest


def main() -> None:
    if not (
        ROOT / "configs" / "datasets"
    ).is_dir():
        raise RuntimeError(
            "Run this script from the repository root."
        )

    checksum_lines = []
    manifests = {}

    for dataset, expected in DATASETS.items():
        checksum_line, manifest = package_dataset(
            dataset,
            expected,
        )

        checksum_lines.append(checksum_line)
        manifests[dataset] = manifest

    checksum_path = (
        OUTPUT_DIR
        / "SHA256SUMS"
    )

    checksum_path.write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
    )

    combined_manifest_path = (
        OUTPUT_DIR
        / "manifest.json"
    )

    combined_manifest_path.write_text(
        json.dumps(
            manifests,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("All archives created successfully.")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Checksums:        {checksum_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
