from pathlib import Path

import pandas as pd


def validate_numeric_dataframe(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Dataframe is empty.")

    non_numeric = [
        col for col in df.columns
        if not pd.api.types.is_numeric_dtype(df[col])
    ]

    if non_numeric:
        raise ValueError(f"Non-numeric columns found: {non_numeric}")

    if df.isna().any().any():
        missing = df.isna().sum()
        raise ValueError(f"Missing values found:\n{missing[missing > 0]}")


def load_tabular_dataset(config: dict) -> pd.DataFrame:
    path = Path(config["path"])
    file_type = config["file_type"]

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if file_type == "whitespace_csv":
        df = pd.read_csv(path, sep=r"\s+")
    elif file_type == "csv":
        df = pd.read_csv(path)
    elif file_type == "tsv":
        df = pd.read_csv(path, sep="\t")
    else:
        raise ValueError(f"Unsupported file_type: {file_type}")

    drop_columns = config["data"].get("drop_columns", [])
    if drop_columns:
        df = df.drop(columns=drop_columns)

    continuous_columns = config["data"].get("continuous_columns", "all")

    if continuous_columns != "all":
        df = df[continuous_columns]

    missing_policy = config["data"].get("missing_policy", "error")

    if missing_policy == "error":
        validate_numeric_dataframe(df)
    elif missing_policy == "drop_rows":
        df = df.dropna()
        validate_numeric_dataframe(df)
    else:
        raise ValueError(f"Unsupported missing_policy: {missing_policy}")

    return df


def make_dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[col].dtype) for col in df.columns],
        "n_rows": [len(df)] * len(df.columns),
        "n_missing": [df[col].isna().sum() for col in df.columns],
        "min": [df[col].min() for col in df.columns],
        "q05": [df[col].quantile(0.05) for col in df.columns],
        "median": [df[col].median() for col in df.columns],
        "mean": [df[col].mean() for col in df.columns],
        "q95": [df[col].quantile(0.95) for col in df.columns],
        "max": [df[col].max() for col in df.columns],
        "std": [df[col].std() for col in df.columns],
    })