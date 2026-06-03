from pathlib import Path
import sys

# Make imports work when running from project root:
# python scripts/build_sachs_marginals.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from copula_sachs.data import load_sachs_data, make_dataset_summary
from copula_sachs.marginals import (
    make_pseudo_observations,
    EmpiricalMarginalLibrary,
)


def main() -> None:
    data_path = PROJECT_ROOT / "data" / "sachs.data.txt"

    artifacts_dir = PROJECT_ROOT / "artifacts"
    reports_tables_dir = PROJECT_ROOT / "reports" / "tables"

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_tables_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Sachs data...")
    df = load_sachs_data(data_path)

    print(f"Loaded data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    print("Creating dataset summary...")
    summary = make_dataset_summary(df)
    summary_path = reports_tables_dir / "sachs_dataset_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("Creating pseudo-observations...")
    u_df = make_pseudo_observations(df, clip_eps=1e-6)
    pseudo_path = artifacts_dir / "sachs_pseudo_observations.csv"
    u_df.to_csv(pseudo_path, index=False)

    print("Fitting empirical marginal library...")
    marginal_library = EmpiricalMarginalLibrary.fit(
        df=df,
        dataset_id="sachs",
        clip_eps=1e-6,
    )

    marginal_path = artifacts_dir / "sachs_marginals_v1.pkl"
    marginal_library.save(marginal_path)

    print("Checking inverse transform reconstruction behavior...")
    reconstructed_df = marginal_library.inverse_transform(u_df)

    reconstruction_summary_path = reports_tables_dir / "sachs_reconstructed_summary.csv"
    reconstructed_summary = make_dataset_summary(reconstructed_df)
    reconstructed_summary.to_csv(reconstruction_summary_path, index=False)

    print("\nT4b Sachs marginal handling complete.")
    print(f"Dataset summary saved to: {summary_path}")
    print(f"Pseudo-observations saved to: {pseudo_path}")
    print(f"Marginal library saved to: {marginal_path}")
    print(f"Reconstructed summary saved to: {reconstruction_summary_path}")


if __name__ == "__main__":
    main()