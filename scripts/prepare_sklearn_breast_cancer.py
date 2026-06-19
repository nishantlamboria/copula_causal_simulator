from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
from sklearn.datasets import load_breast_cancer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = PROJECT_ROOT / "data" / "breast_cancer.csv"
METADATA_PATH = PROJECT_ROOT / "data" / "breast_cancer_metadata.json"


# Twelve continuous predictors selected from different feature groups.
# The diagnosis target is intentionally excluded.
SELECTED_FEATURES = [
    "mean radius",
    "mean texture",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
    "radius error",
    "texture error",
    "worst radius",
    "worst texture",
]


def make_safe_column_name(name: str) -> str:
    """
    Convert scikit-learn feature names into simple column names suitable
    for configuration files, artifact names, and generated model filenames.
    """
    return (
        name.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
    )


def main() -> None:
    dataset = load_breast_cancer(as_frame=True)

    dataset_frame = getattr(dataset, "frame", None)
    if dataset_frame is None:
        raise RuntimeError(
            "scikit-learn did not return the breast cancer dataset as a DataFrame."
        )

    source_df = dataset_frame.copy()

    missing_features = [
        feature
        for feature in SELECTED_FEATURES
        if feature not in source_df.columns
    ]

    if missing_features:
        raise KeyError(
            "The following expected breast cancer features were not found:\n"
            f"{missing_features}\n\n"
            f"Available columns:\n{source_df.columns.tolist()}"
        )

    # Keep predictors only. The target column is not included.
    output_df = source_df[SELECTED_FEATURES].copy()

    rename_mapping = {
        column: make_safe_column_name(column)
        for column in output_df.columns
    }

    output_df = output_df.rename(columns=rename_mapping)

    # Ensure all retained columns are numeric.
    for column in output_df.columns:
        output_df[column] = pd.to_numeric(
            output_df[column],
            errors="raise",
        )

    if output_df.isna().any().any():
        missing_counts = output_df.isna().sum()
        missing_counts = missing_counts[missing_counts > 0]

        raise ValueError(
            "Missing values were found in the prepared dataset:\n"
            f"{missing_counts}"
        )

    if output_df.empty:
        raise ValueError("Prepared breast cancer dataset is empty.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)

    metadata = {
        "dataset_id": "breast_cancer",
        "source": "sklearn.datasets.load_breast_cancer",
        "description": (
            "Twelve-feature continuous subset of the Wisconsin Diagnostic "
            "Breast Cancer dataset. The diagnosis target is excluded."
        ),
        "num_rows": int(output_df.shape[0]),
        "num_columns": int(output_df.shape[1]),
        "original_selected_features": SELECTED_FEATURES,
        "saved_columns": output_df.columns.tolist(),
        "target_included": False,
        "output_file": str(OUTPUT_PATH.relative_to(PROJECT_ROOT)),
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("Breast cancer dataset prepared successfully.")
    print(f"Shape: {output_df.shape}")
    print(f"Saved dataset: {OUTPUT_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")
    print("\nColumns:")
    for column in output_df.columns:
        print(f"  - {column}")

    print("\nPreview:")
    print(output_df.head())


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"\nFailed to prepare breast cancer dataset:\n{exc}",
            file=sys.stderr,
        )
        raise