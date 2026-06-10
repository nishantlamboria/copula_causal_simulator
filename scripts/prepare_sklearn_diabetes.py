from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.datasets import load_diabetes


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    output_path = PROJECT_ROOT / "data" / "diabetes.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = load_diabetes(as_frame=True)

    # sklearn may return a Bunch or a tuple; normalize to a Bunch-like object
    # Use a dynamic type to satisfy static analyzers.
    bunch: Any
    if hasattr(result, "data"):
        bunch = result
    else:
        # assume (bunch, ...) or (data, target)
        bunch = result[0]

    # Use only the 10 continuous covariates.
    # We exclude the target because the coupling library currently treats all columns symmetrically.
    df = bunch.data.copy()

    df.to_csv(output_path, index=False)

    print(f"Saved diabetes dataset to: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()