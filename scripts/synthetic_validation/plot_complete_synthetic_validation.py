"""Create publication-ready plots and a Markdown report for complete validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot complete synthetic-validation results.")
    parser.add_argument(
        "--results-directory",
        required=True,
        help="Directory produced by run_complete_synthetic_validation.py.",
    )
    parser.add_argument(
        "--report-directory",
        default="reports/synthetic_validation/complete",
        help="Output report directory.",
    )
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def save_figure(figure: plt.Figure, path_without_suffix: Path) -> None:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path_without_suffix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(path_without_suffix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_marginal_summary(data: pd.DataFrame, path: Path) -> None:
    families = ["gaussian", "student", "clayton", "gumbel", "frank"]
    accuracy = (
        data.groupby("true_family")["family_correct"].mean().reindex(families).dropna()
    )
    errors = data.groupby("true_family")["mean_marginal_error"].mean().reindex(accuracy.index)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].bar(accuracy.index, accuracy.to_numpy())
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Family recovery accuracy")
    axes[0].set_title("Copula recovery after empirical marginals")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(errors.index, errors.to_numpy())
    axes[1].set_ylabel("Mean normalized quantile error")
    axes[1].set_title("Generated marginal fidelity")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle("End-to-end marginal transformation validation")
    figure.tight_layout()
    save_figure(figure, path)


def plot_dvine_summary(data: pd.DataFrame, path: Path) -> None:
    metrics = pd.Series(
        {
            "Order selected": data["generating_order_selected"].mean(),
            "C(P1,P2) family": data["true_order_first_middle_correct"].mean(),
            "C(P2,X) family": data["true_order_middle_child_correct"].mean(),
            "C(P1,X|P2) family": data["true_order_conditional_correct"].mean(),
        }
    )
    scenario_metrics = data.groupby("scenario_id").agg(
        kendall_mae=("pairwise_kendall_mae", "mean"),
        loglik_gap=("test_loglik_gap_from_oracle", "mean"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    axes[0].barh(metrics.index, metrics.to_numpy())
    axes[0].set_xlim(0.0, 1.05)
    axes[0].set_xlabel("Recovery rate")
    axes[0].set_title("Known mechanism recovery")
    axes[0].grid(axis="x", alpha=0.25)
    x = np.arange(len(scenario_metrics))
    axes[1].bar(x - 0.18, scenario_metrics["kendall_mae"], width=0.36, label="Kendall MAE")
    axes[1].bar(
        x + 0.18,
        np.abs(scenario_metrics["loglik_gap"]),
        width=0.36,
        label="|Loglik gap from oracle|",
    )
    axes[1].set_xticks(x, scenario_metrics.index, rotation=25, ha="right")
    axes[1].set_title("Generated three-dimensional fit")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle("Known simplified D-vine validation")
    figure.tight_layout()
    save_figure(figure, path)


def plot_full_dag_summary(data: pd.DataFrame, path: Path) -> None:
    metric_names = [
        "Pairwise Kendall MAE",
        "5D energy distance",
        "5D MMD²",
        "Marginal quantile error",
    ]
    values = [
        data["pairwise_kendall_mae"].mean(),
        data["energy_distance_5d"].mean(),
        data["mmd2_5d"].mean(),
        data["mean_marginal_quantile_error"].mean(),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].barh(metric_names, values)
    axes[0].set_title("Joint and marginal discrepancies")
    axes[0].grid(axis="x", alpha=0.25)
    recovery = pd.Series(
        {
            "Two-parent order": data["generating_x3_order_selected"].mean(),
            "X1→X2 family": data["x2_family_correct"].mean(),
            "X3→X4 family": data["x4_family_correct"].mean(),
            "X2→X5 family": data["x5_family_correct"].mean(),
        }
    )
    axes[1].barh(recovery.index, recovery.to_numpy())
    axes[1].set_xlim(0.0, 1.05)
    axes[1].set_title("Local mechanism recovery")
    axes[1].set_xlabel("Recovery rate")
    axes[1].grid(axis="x", alpha=0.25)
    figure.suptitle("Complete five-node DAG validation")
    figure.tight_layout()
    save_figure(figure, path)


def plot_non_simplified_curve(curves: pd.DataFrame, path: Path) -> None:
    grouped = curves.groupby("conditioning_u2").agg(
        true_tau=("true_tau", "mean"),
        simplified_tau=("simplified_tau", "mean"),
        binned_tau=("binned_tau", "mean"),
        binned_low=("binned_tau", lambda values: np.quantile(values, 0.1)),
        binned_high=("binned_tau", lambda values: np.quantile(values, 0.9)),
    )
    figure, axis = plt.subplots(figsize=(7.5, 4.6))
    x = grouped.index.to_numpy()
    axis.plot(x, grouped["true_tau"], linewidth=2.3, label="True conditional tau")
    axis.plot(x, grouped["simplified_tau"], linestyle="--", linewidth=2.0, label="Simplified fit")
    axis.step(x, grouped["binned_tau"], where="mid", linewidth=2.0, label="Adaptive binned fit")
    axis.fill_between(x, grouped["binned_low"], grouped["binned_high"], alpha=0.18)
    axis.set_xlabel("Conditioning quantile $u_2$")
    axis.set_ylabel("Conditional Kendall's tau")
    axis.set_title("Relaxing the simplifying assumption")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    save_figure(figure, path)


def plot_non_simplified_metrics(data: pd.DataFrame, path: Path) -> None:
    labels = ["Test log-likelihood", "Conditional tau RMSE"]
    simplified = [
        data["simplified_test_loglik_mean"].mean(),
        data["simplified_tau_rmse"].mean(),
    ]
    binned = [
        data["binned_test_loglik_mean"].mean(),
        data["binned_tau_rmse"].mean(),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4.1))
    axes[0].bar(["Simplified", "Adaptive binned"], [simplified[0], binned[0]])
    axes[0].set_title(labels[0])
    axes[0].set_ylabel("Mean log density per observation")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(["Simplified", "Adaptive binned"], [simplified[1], binned[1]])
    axes[1].set_title(labels[1])
    axes[1].set_ylabel("RMSE")
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle("Simplified versus data-adaptive conditional copulas")
    figure.tight_layout()
    save_figure(figure, path)


def write_report(
    report_directory: Path,
    summary: dict,
    marginal: pd.DataFrame,
    dvine: pd.DataFrame,
    dag: pd.DataFrame,
    non_simplified: pd.DataFrame,
) -> None:
    report_directory.mkdir(parents=True, exist_ok=True)
    pair = summary["pair_family_validation"]
    text = f"""# Complete synthetic validation

This report validates the simulator against known data-generating mechanisms.
The compact suite is a fast, reproducible verification run; the included full
configuration increases seeds and sample-size settings for final reporting.

## 1. Bivariate pair-copula calibration

- Successful runs: **{pair.get('successful_runs', 'not available')}**
- Exact family recovery: **{pair.get('family_recovery_accuracy', float('nan')):.1%}**
- Mean absolute Kendall-tau error: **{pair.get('mean_tau_abs_error', float('nan')):.4f}**

The main residual ambiguity is between high-degree-of-freedom Student copulas
and Gaussian copulas. Dependence strength is nevertheless recovered accurately.

## 2. Empirical marginals and inverse transformation

- Runs: **{len(marginal)}**
- Family recovery: **{marginal['family_correct'].mean():.1%}**
- Mean absolute Kendall-tau error: **{marginal['tau_abs_error'].mean():.4f}**
- Mean normalized marginal quantile error: **{marginal['mean_marginal_error'].mean():.4f}**
- Maximum rank-invariance discrepancy: **{marginal['max_rank_invariance_error'].max():.2e}**

Strictly monotone marginal transformations preserve ranks exactly. The experiment
therefore confirms that the empirical pseudo-observation layer does not alter
the copula information in continuous data, while empirical inverse marginals
retain small finite-sample quantile error.

## 3. Known simplified D-vines and flexible parent order

- Runs: **{len(dvine)}**
- Generating order selected: **{dvine['generating_order_selected'].mean():.1%}**
- First pair-family recovery: **{dvine['true_order_first_middle_correct'].mean():.1%}**
- Second pair-family recovery: **{dvine['true_order_middle_child_correct'].mean():.1%}**
- Conditional family recovery: **{dvine['true_order_conditional_correct'].mean():.1%}**
- Mean three-dimensional Kendall error: **{dvine['pairwise_kendall_mae'].mean():.4f}**
- Mean conditional log-likelihood gap from oracle: **{dvine['test_loglik_gap_from_oracle'].mean():.4f}**

The parent order is selected using held-out conditional log-likelihood rather
than arbitrary input order. In three dimensions, order should be interpreted as
a predictive representation choice rather than a universally identifiable
causal object.

## 4. Complete known DAG

The validation DAG is

`X1 → X2`, `X1 → X3`, `X2 → X3`, `X3 → X4`, `X2 → X5`.

It contains a root, one-parent mechanisms, a two-parent mechanism, and multiple
topological generations.

- Runs: **{len(dag)}**
- Two-parent generating order selected: **{dag['generating_x3_order_selected'].mean():.1%}**
- One-parent family recovery: **{(dag['x2_family_correct'].mean() + dag['x4_family_correct'].mean() + dag['x5_family_correct'].mean()) / 3.0:.1%}**
- Mean pairwise Kendall error: **{dag['pairwise_kendall_mae'].mean():.4f}**
- Mean five-dimensional energy distance: **{dag['energy_distance_5d'].mean():.4f}**
- Mean marginal quantile error: **{dag['mean_marginal_quantile_error'].mean():.4f}**

This is the central end-to-end check: data are generated from a known DAG,
calibrated from observations, and regenerated using fitted local mechanisms.

## 5. Violation of the simplifying assumption

The true conditional Gaussian-copula parameter varies smoothly with the
conditioning quantile. A single conditional copula is compared with a
cross-validated equal-frequency binned model.

- Runs: **{len(non_simplified)}**
- Mean selected number of bins: **{non_simplified['selected_bins'].mean():.2f}**
- Mean held-out log-likelihood improvement: **{non_simplified['binned_test_loglik_improvement'].mean():.4f}**
- Simplified conditional-tau RMSE: **{non_simplified['simplified_tau_rmse'].mean():.4f}**
- Adaptive binned conditional-tau RMSE: **{non_simplified['binned_tau_rmse'].mean():.4f}**

The binned model is a useful data-driven baseline and diagnostic. It reduces
conditional-tau error and improves held-out likelihood in this controlled
non-simplified setting. For higher-dimensional conditioning sets, smooth
varying-parameter models or recursive partitioning should replace naive
multidimensional binning.

## Figures

- `figures/marginal_validation_summary.*`
- `figures/dvine_validation_summary.*`
- `figures/full_dag_validation_summary.*`
- `figures/non_simplified_tau_curve.*`
- `figures/non_simplified_metrics.*`

## Scope and limitations

The synthetic suite establishes recovery under mechanisms that belong to, or
are deliberately close to, the implemented model class. It does not by itself
establish causal validity on observational real-world data. The full
configuration should be run for final uncertainty estimates; the compact run
primarily verifies correctness and produces immediate report-ready evidence.
"""
    (report_directory / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_directory = resolve(args.results_directory)
    report_directory = resolve(args.report_directory)
    figures = report_directory / "figures"
    tables = report_directory / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    marginal = pd.read_csv(results_directory / "marginal_validation_results.csv")
    dvine = pd.read_csv(results_directory / "dvine_validation_results.csv")
    dag = pd.read_csv(results_directory / "full_dag_validation_results.csv")
    non_simplified = pd.read_csv(results_directory / "non_simplified_validation_results.csv")
    curves = pd.read_csv(results_directory / "non_simplified_tau_curves.csv")
    with (results_directory / "synthetic_validation_summary.json").open(
        "r", encoding="utf-8"
    ) as handle:
        summary = json.load(handle)

    plot_marginal_summary(marginal, figures / "marginal_validation_summary")
    plot_dvine_summary(dvine, figures / "dvine_validation_summary")
    plot_full_dag_summary(dag, figures / "full_dag_validation_summary")
    plot_non_simplified_curve(curves, figures / "non_simplified_tau_curve")
    plot_non_simplified_metrics(non_simplified, figures / "non_simplified_metrics")

    marginal.groupby(["true_family", "marginal_1", "marginal_2"], as_index=False).agg(
        runs=("seed", "count"),
        family_recovery=("family_correct", "mean"),
        tau_abs_error=("tau_abs_error", "mean"),
        marginal_error=("mean_marginal_error", "mean"),
    ).to_csv(tables / "marginal_summary.csv", index=False)
    dvine.groupby("scenario_id", as_index=False).agg(
        runs=("seed", "count"),
        order_selection=("generating_order_selected", "mean"),
        conditional_family_recovery=("true_order_conditional_correct", "mean"),
        kendall_mae=("pairwise_kendall_mae", "mean"),
        energy_distance=("energy_distance_3d", "mean"),
    ).to_csv(tables / "dvine_summary.csv", index=False)
    dag.to_csv(tables / "full_dag_runs.csv", index=False)
    non_simplified.to_csv(tables / "non_simplified_runs.csv", index=False)
    (tables / "headline_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_report(report_directory, summary, marginal, dvine, dag, non_simplified)
    print(f"Report written to: {report_directory}")


if __name__ == "__main__":
    main()
