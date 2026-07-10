"""Create diagnostics for adaptive conditional-copula selection.

The input is the conditional-vine summary CSV written by
``scripts/build_conditional_vine_library.py``.  The script produces compact
CSV tables, aggregate figures, per-mechanism conditional-tau plots, and a
Markdown summary.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report adaptive conditional-copula model selection."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-mechanism-plots", type=int, default=25)
    return parser.parse_args()


def parse_literal(value: Any, default: Any) -> Any:
    if isinstance(value, type(default)):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return default
    return parsed if isinstance(parsed, type(default)) else default


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


def save_figure(fig: plt.Figure, path_without_suffix: Path) -> None:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_without_suffix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path_without_suffix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.summary.exists():
        raise FileNotFoundError(args.summary)
    if args.max_mechanism_plots < 0:
        raise ValueError("--max-mechanism-plots must be non-negative.")

    output_dir = args.output_dir
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.summary)
    required = {
        "child",
        "parent_set",
        "status",
        "conditional_model_type",
        "selected_number_bins",
        "simplified_validation_score",
        "selected_validation_score",
        "score_improvement_over_simplified",
        "conditional_family",
        "conditional_tau_diagnostics",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            "Summary CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    successful = data.loc[data["status"] == "ok"].copy()
    if successful.empty:
        raise RuntimeError("The summary contains no successful mechanisms.")

    selected_columns = [
        "child",
        "parent_set",
        "selected_order",
        "conditional_model_type",
        "selected_number_bins",
        "simplified_validation_score",
        "selected_validation_score",
        "score_improvement_over_simplified",
        "conditional_family",
        "conditional_rotation",
        "conditional_bin_counts",
    ]
    selected_columns = [col for col in selected_columns if col in successful]
    successful[selected_columns].to_csv(
        tables_dir / "conditional_model_selection_summary.csv",
        index=False,
    )

    counts = (
        successful.groupby(
            ["conditional_model_type", "selected_number_bins"],
            dropna=False,
            as_index=False,
        )
        .size()
        .rename(columns={"size": "number_mechanisms"})
    )
    counts.to_csv(tables_dir / "selected_bin_count_distribution.csv", index=False)

    diagnostics_rows: list[dict[str, Any]] = []
    for _, row in successful.iterrows():
        diagnostics = parse_literal(row["conditional_tau_diagnostics"], [])
        for diagnostic in diagnostics:
            if not isinstance(diagnostic, dict):
                continue
            diagnostics_rows.append(
                {
                    "child": row["child"],
                    "parent_set": row["parent_set"],
                    "selected_order": row.get("selected_order", ""),
                    "selected_number_bins": int(row["selected_number_bins"]),
                    **diagnostic,
                }
            )
    diagnostics_df = pd.DataFrame(diagnostics_rows)
    diagnostics_df.to_csv(
        tables_dir / "conditional_tau_by_bin.csv",
        index=False,
    )

    distribution = counts.groupby("selected_number_bins")["number_mechanisms"].sum()
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.bar([str(int(value)) for value in distribution.index], distribution.values)
    ax.set_xlabel("Selected number of conditional bins")
    ax.set_ylabel("Number of local mechanisms")
    ax.set_title("Adaptive conditional-copula model selection")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, figures_dir / "selected_bin_count_distribution")

    improvement = pd.to_numeric(
        successful["score_improvement_over_simplified"], errors="coerce"
    )
    finite_mask = np.isfinite(improvement.to_numpy(dtype=float))
    if np.any(finite_mask):
        plot_data = successful.loc[finite_mask].copy().reset_index(drop=True)
        plot_values = improvement.loc[finite_mask].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(9.0, 4.8))
        ax.scatter(np.arange(len(plot_values)), plot_values, s=24)
        ax.axhline(0.0, linewidth=1.0)
        ax.set_xlabel("Two-parent local mechanism")
        ax.set_ylabel("Held-out log-likelihood improvement")
        ax.set_title("Selected conditional model versus simplifying assumption")
        ax.grid(axis="y", alpha=0.25)
        save_figure(fig, figures_dir / "score_improvement_over_simplified")

    if not diagnostics_df.empty and args.max_mechanism_plots > 0:
        grouped = diagnostics_df.groupby(["child", "parent_set"], sort=True)
        for plot_index, ((child, parent_set), group) in enumerate(grouped):
            if plot_index >= args.max_mechanism_plots:
                break
            group = group.sort_values("bin_center")
            fig, ax = plt.subplots(figsize=(7.0, 4.5))
            ax.plot(
                group["bin_center"],
                group["empirical_conditional_tau"],
                marker="o",
                label="Empirical binwise tau",
            )
            ax.plot(
                group["bin_center"],
                group["fitted_conditional_tau"],
                marker="s",
                label="Fitted binwise tau",
            )
            global_tau = group["global_empirical_conditional_tau"].iloc[0]
            ax.axhline(global_tau, linestyle="--", label="Global empirical tau")
            ax.set_xlim(0.0, 1.0)
            ax.set_xlabel("Conditioning-parent copula quantile")
            ax.set_ylabel("Conditional Kendall's tau")
            ax.set_title(f"{child} given {parent_set}")
            ax.grid(alpha=0.25)
            ax.legend()
            filename = safe_name(f"tau_curve__{child}__{parent_set}")
            save_figure(fig, figures_dir / filename)

    n_total = len(successful)
    n_binned = int((successful["selected_number_bins"] > 1).sum())
    median_improvement = float(np.nanmedian(improvement)) if finite_mask.any() else float("nan")
    readme = f"""# Adaptive conditional-copula diagnostics

Input: `{args.summary}`

- Successful two-parent mechanisms: **{n_total}**
- Mechanisms selecting more than one bin: **{n_binned}** ({n_binned / n_total:.1%})
- Mechanisms retaining the simplifying assumption: **{n_total - n_binned}**
- Median raw best-score improvement over the simplified model: **{median_improvement:.6f}**

A multi-bin model is selected only when its held-out mean conditional
log-likelihood exceeds the simplified model by the configured minimum
improvement.  Non-edge or causal interpretations should not be attached to
this diagnostic: it concerns only the statistical conditional-copula
representation for a fixed DAG parent set and selected D-vine order.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"Report written to: {output_dir}")


if __name__ == "__main__":
    main()
