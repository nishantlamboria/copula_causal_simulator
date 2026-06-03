from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    summary_path = PROJECT_ROOT / "reports" / "tables" / "sachs_paircopula_summary.csv"
    output_dir = PROJECT_ROOT / "reports" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Could not find pair-copula summary file: {summary_path}\n"
            "Run scripts/build_sachs_paircopula_library.py first."
        )

    df = pd.read_csv(summary_path)

    print("Loaded pair-copula summary:")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # ------------------------------------------------------------------
    # 1. Family counts
    # ------------------------------------------------------------------
    family_counts = (
        df["family"]
        .value_counts()
        .rename_axis("family")
        .reset_index(name="count")
    )

    family_counts_path = output_dir / "sachs_family_counts.csv"
    family_counts.to_csv(family_counts_path, index=False)

    print("\nSelected family counts:")
    print(family_counts)

    # ------------------------------------------------------------------
    # 2. Non-independent selected pairs
    # ------------------------------------------------------------------
    non_indep = df[df["family"] != "indep"].copy()

    display_cols = [
        "var1",
        "var2",
        "family",
        "rotation",
        "parameters",
        "empirical_kendall_tau",
        "empirical_spearman_rho",
        "empirical_lower_tail_q05",
        "empirical_upper_tail_q05",
        "loglik",
        "aic",
        "bic",
    ]

    existing_display_cols = [col for col in display_cols if col in non_indep.columns]

    non_indep = non_indep[existing_display_cols].sort_values(
        by="bic",
        ascending=True,
        na_position="last",
    )

    non_indep_path = output_dir / "sachs_non_independent_pairs.csv"
    non_indep.to_csv(non_indep_path, index=False)

    print("\nNon-independent selected pairs:")
    if len(non_indep) == 0:
        print("No non-independent pairs selected.")
    else:
        print(non_indep.to_string(index=False))

    # ------------------------------------------------------------------
    # 3. Top empirical Kendall pairs
    # ------------------------------------------------------------------
    df["abs_empirical_kendall_tau"] = df["empirical_kendall_tau"].abs()

    top_kendall = df.sort_values(
        by="abs_empirical_kendall_tau",
        ascending=False,
        na_position="last",
    ).copy()

    top_cols = [
        "var1",
        "var2",
        "family",
        "empirical_kendall_tau",
        "abs_empirical_kendall_tau",
        "empirical_spearman_rho",
        "empirical_lower_tail_q05",
        "empirical_upper_tail_q05",
        "aic",
        "bic",
    ]

    existing_top_cols = [col for col in top_cols if col in top_kendall.columns]

    top_kendall = top_kendall[existing_top_cols]

    top_kendall_path = output_dir / "sachs_top_kendall_pairs.csv"
    top_kendall.to_csv(top_kendall_path, index=False)

    print("\nTop 15 pairs by absolute empirical Kendall tau:")
    print(top_kendall.head(15).to_string(index=False))

    # ------------------------------------------------------------------
    # 4. Check: strong empirical tau but selected as independence
    # ------------------------------------------------------------------
    suspicious = df[
        (df["family"] == "indep")
        & (df["abs_empirical_kendall_tau"] >= 0.10)
    ].copy()

    suspicious = suspicious.sort_values(
        by="abs_empirical_kendall_tau",
        ascending=False,
    )

    suspicious_path = output_dir / "sachs_indep_but_tau_ge_010.csv"
    suspicious.to_csv(suspicious_path, index=False)

    print("\nPairs selected as independence but with |Kendall tau| >= 0.10:")
    if len(suspicious) == 0:
        print("None.")
    else:
        print(
            suspicious[
                [
                    "var1",
                    "var2",
                    "family",
                    "empirical_kendall_tau",
                    "empirical_spearman_rho",
                    "aic",
                    "bic",
                ]
            ].to_string(index=False)
        )

    print("\nSaved analysis files:")
    print(f"- {family_counts_path}")
    print(f"- {non_indep_path}")
    print(f"- {top_kendall_path}")
    print(f"- {suspicious_path}")


if __name__ == "__main__":
    main()