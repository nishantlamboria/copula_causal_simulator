from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from copula_sachs.data import load_sachs_data
from copula_sachs.marginals import make_pseudo_observations
from copula_sachs.paircopula_library import build_paircopula_library


def main() -> None:
    data_path = PROJECT_ROOT / "data" / "sachs.data.txt"

    artifacts_dir = PROJECT_ROOT / "artifacts"
    tables_dir = PROJECT_ROOT / "reports" / "tables"
    model_dir = artifacts_dir / "paircopula_models_sachs_v1"

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Sachs data...")
    df = load_sachs_data(data_path)

    print(f"Loaded data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    print("Creating pseudo-observations...")
    u_df = make_pseudo_observations(df, clip_eps=1e-6)

    print("Building Sachs pair-copula library...")
    library = build_paircopula_library(
        u_df=u_df,
        dataset_id="sachs",
        selection_criterion="bic",
        allow_rotations=True,
        num_threads=1,
        model_dir=model_dir,
    )

    summary_df = library.to_dataframe()

    summary_path = tables_dir / "sachs_paircopula_summary.csv"
    library_path = artifacts_dir / "coupling_library_sachs_v1.pkl"

    summary_df.to_csv(summary_path, index=False)
    library.save_pickle(library_path)

    n_ok = int((summary_df["status"] == "ok").sum())
    n_failed = int((summary_df["status"] == "failed").sum())

    print("\nT4c Sachs pair-copula library complete.")
    print(f"Total pair models: {len(summary_df)}")
    print(f"Successful fits: {n_ok}")
    print(f"Failed fits: {n_failed}")
    print(f"Summary saved to: {summary_path}")
    print(f"Library metadata saved to: {library_path}")
    print(f"Individual Bicop JSON models saved in: {model_dir}")

    print("\nSelected family counts:")
    print(summary_df["family"].value_counts())


if __name__ == "__main__":
    main()