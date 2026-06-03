from pathlib import Path

import pandas as pd


def validate_numeric_dataframe(df: pd.DataFrame) -> None:
    """
    Validate that the dataframe is non-empty, numeric, and contains no missing values.
    """
    if df.empty:
        raise ValueError("The dataframe is empty.")

    non_numeric_columns = [
        col for col in df.columns
        if not pd.api.types.is_numeric_dtype(df[col])
    ]

    if non_numeric_columns:
        raise ValueError(f"Non-numeric columns found: {non_numeric_columns}")

    if df.isna().any().any():
        missing = df.isna().sum()
        raise ValueError(f"Missing values found:\n{missing[missing > 0]}")


def load_sachs_data(path: str | Path) -> pd.DataFrame:
    """
    Load the Sachs dataset from a whitespace-separated text file.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Could not find Sachs data file: {path}")

    df = pd.read_csv(path, sep=r"\s+")
    validate_numeric_dataframe(df)

    return df


def make_dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a compact dataset summary table.
    """
    summary = pd.DataFrame({
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

    return summary