"""Create publication-ready reports for pair-copula recovery experiments.

The script consumes the detailed result CSV produced by
``run_pair_family_recovery.py`` and creates PNG/PDF figures, compact CSV
summaries, and a Markdown interpretation report.

Example
-------
Run from the project root::

    python scripts/synthetic_validation/plot_pair_recovery_results.py ^
        --results outputs/synthetic_validation/pair_family_recovery_full/\
        pair_family_recovery_results.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIRECTORY = PROJECT_ROOT / "src"

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from copula_causal_sim.synthetic.pair_recovery_reporting import (  # noqa: E402
    DEFAULT_REPRESENTATIVE_CASES,
    FAMILY_ORDER,
    RepresentativeCase,
    confusion_tables,
    density_grid,
    family_label,
    family_sample_summary,
    ordered_families,
    read_results,
    reconstruct_model,
    scenario_summary,
    select_representative_run,
    summary_statistics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create figures and summary tables for a synthetic "
            "pair-copula recovery benchmark."
        )
    )
    parser.add_argument(
        "--results",
        required=True,
        type=str,
        help="Path to pair_family_recovery_results.csv.",
    )
    parser.add_argument(
        "--output-directory",
        default=(
            "reports/synthetic_validation/pair_family_recovery"
        ),
        type=str,
        help="Directory for generated figures, tables, and report.",
    )
    parser.add_argument(
        "--grid-size",
        default=140,
        type=int,
        help="Grid resolution for copula-density figures.",
    )
    parser.add_argument(
        "--scatter-points",
        default=1000,
        type=int,
        help="Maximum pseudo-observations shown per scatter panel.",
    )
    return parser.parse_args()


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def configure_matplotlib() -> None:
    """Apply restrained, portable plotting defaults."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def family_colors(families: Iterable[str]) -> dict[str, str]:
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ordered = ordered_families(families)
    return {
        family: cycle[index % len(cycle)]
        for index, family in enumerate(ordered)
    }


def save_figure(
    figure: plt.Figure,
    figures_directory: Path,
    stem: str,
) -> None:
    figures_directory.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        figure.savefig(
            figures_directory / f"{stem}.{extension}",
            bbox_inches="tight",
        )
    plt.close(figure)


def annotate_matrix(
    axis: plt.Axes,
    matrix: np.ndarray,
    *,
    formatter,
) -> None:
    threshold = float(np.nanmax(matrix)) * 0.55
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            text_color = "white" if value > threshold else "black"
            axis.text(
                column,
                row,
                formatter(value),
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )


def plot_confusion_matrix(
    matrix: pd.DataFrame,
    *,
    normalized: bool,
    figures_directory: Path,
) -> None:
    values = matrix.to_numpy(dtype=float)
    figure, axis = plt.subplots(figsize=(7.2, 6.0))
    image = axis.imshow(
        values,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0 if normalized else None,
        aspect="equal",
    )

    labels = [family_label(value) for value in matrix.index]
    axis.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
    axis.set_yticks(np.arange(len(labels)), labels)
    axis.set_xlabel("Selected family")
    axis.set_ylabel("True family")

    if normalized:
        axis.set_title("Pair-copula family recovery: row proportions")
        annotate_matrix(
            axis,
            values,
            formatter=lambda value: f"{100.0 * value:.1f}%",
        )
        colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        stem = "confusion_matrix_row_proportions"
    else:
        axis.set_title("Pair-copula family recovery: counts")
        annotate_matrix(
            axis,
            values,
            formatter=lambda value: f"{int(value)}",
        )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        stem = "confusion_matrix_counts"

    figure.tight_layout()
    save_figure(figure, figures_directory, stem)


def plot_family_recovery_by_sample_size(
    summary: pd.DataFrame,
    *,
    figures_directory: Path,
) -> None:
    families = ordered_families(summary["true_family"])
    colors = family_colors(families)
    figure, axis = plt.subplots(figsize=(8.0, 5.0))

    for family in families:
        subset = summary.loc[summary["true_family"] == family]
        axis.plot(
            subset["sample_size"],
            subset["family_recovery_accuracy"],
            marker="o",
            linewidth=2,
            markersize=5,
            label=family_label(family),
            color=colors[family],
        )

    axis.set_xlabel("Sample size")
    axis.set_ylabel("Exact family recovery")
    axis.set_ylim(-0.02, 1.04)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_xticks(sorted(summary["sample_size"].unique()))
    axis.grid(axis="y", alpha=0.25)
    axis.set_title("Family recovery improves with sample size")
    axis.legend(ncol=3, frameon=False, loc="lower right")
    figure.tight_layout()
    save_figure(
        figure,
        figures_directory,
        "family_recovery_by_sample_size",
    )


def scenario_display_label(row: pd.Series) -> str:
    family = str(row["true_family"])
    if family == "student":
        return f"Student t (df={row['true_student_df']:g})"
    return family_label(family)


def plot_recovery_by_tau(
    scenario_table: pd.DataFrame,
    *,
    figures_directory: Path,
) -> None:
    table = scenario_table.loc[
        scenario_table["true_family"] != "indep"
    ].copy()
    sample_sizes = sorted(table["sample_size"].unique())
    figure, axes = plt.subplots(
        1,
        len(sample_sizes),
        figsize=(5.1 * len(sample_sizes), 4.4),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes)

    table["display_label"] = table.apply(scenario_display_label, axis=1)
    labels = table["display_label"].drop_duplicates().tolist()
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {
        label: cycle[index % len(cycle)]
        for index, label in enumerate(labels)
    }

    for axis, sample_size in zip(axes, sample_sizes):
        subset_n = table.loc[table["sample_size"] == sample_size]
        for label in labels:
            subset = subset_n.loc[
                subset_n["display_label"] == label
            ].sort_values("true_tau")
            if subset.empty:
                continue
            axis.plot(
                subset["true_tau"],
                subset["family_recovery_accuracy"],
                marker="o",
                linewidth=1.8,
                label=label,
                color=colors[label],
            )
        axis.set_title(f"n = {sample_size}")
        axis.set_xlabel("True Kendall's tau")
        axis.set_xticks([0.2, 0.5, 0.8])
        axis.set_ylim(-0.02, 1.04)
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Exact family recovery")
    handles, legend_labels = axes[-1].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
    )
    figure.suptitle(
        "Recovery by dependence strength and sample size",
        y=1.02,
    )
    figure.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    save_figure(figure, figures_directory, "family_recovery_by_tau")


def plot_tau_recovery(
    results: pd.DataFrame,
    *,
    figures_directory: Path,
) -> None:
    sample_sizes = sorted(results["sample_size"].unique())
    families = ordered_families(results["true_family"])
    colors = family_colors(families)
    figure, axes = plt.subplots(
        1,
        len(sample_sizes),
        figsize=(5.0 * len(sample_sizes), 4.6),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes)

    rng = np.random.default_rng(20260709)

    for axis, sample_size in zip(axes, sample_sizes):
        subset_n = results.loc[results["sample_size"] == sample_size]
        for family in families:
            subset = subset_n.loc[subset_n["true_family"] == family]
            if subset.empty:
                continue
            jitter = rng.normal(0.0, 0.008, size=len(subset))
            axis.scatter(
                subset["true_tau"].to_numpy() + jitter,
                subset["estimated_tau"],
                s=14,
                alpha=0.38,
                label=family_label(family),
                color=colors[family],
                edgecolors="none",
            )
        axis.plot(
            [-0.05, 0.85],
            [-0.05, 0.85],
            linestyle="--",
            linewidth=1.2,
            color="black",
        )
        axis.set_title(f"n = {sample_size}")
        axis.set_xlabel("True Kendall's tau")
        axis.set_xlim(-0.05, 0.85)
        axis.set_ylim(-0.05, 0.85)
        axis.grid(alpha=0.2)

    axes[0].set_ylabel("Estimated Kendall's tau")
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    figure.suptitle("Kendall's tau is recovered accurately", y=1.02)
    figure.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
    save_figure(figure, figures_directory, "estimated_vs_true_tau")


def plot_tau_error_by_sample_size(
    summary: pd.DataFrame,
    *,
    figures_directory: Path,
) -> None:
    families = ordered_families(summary["true_family"])
    colors = family_colors(families)
    figure, axis = plt.subplots(figsize=(8.0, 5.0))

    for family in families:
        subset = summary.loc[summary["true_family"] == family]
        axis.plot(
            subset["sample_size"],
            subset["mean_tau_abs_error"],
            marker="o",
            linewidth=2,
            markersize=5,
            label=family_label(family),
            color=colors[family],
        )

    axis.set_xlabel("Sample size")
    axis.set_ylabel("Mean absolute Kendall-tau error")
    axis.set_xticks(sorted(summary["sample_size"].unique()))
    axis.set_ylim(bottom=0.0)
    axis.grid(axis="y", alpha=0.25)
    axis.set_title("Dependence-strength estimation remains accurate")
    axis.legend(ncol=3, frameon=False)
    figure.tight_layout()
    save_figure(
        figure,
        figures_directory,
        "tau_absolute_error_by_sample_size",
    )


def plot_student_recovery(
    results: pd.DataFrame,
    *,
    figures_directory: Path,
) -> None:
    student = results.loc[results["true_family"] == "student"].copy()
    table = (
        student.groupby(
            ["true_student_df", "true_tau", "sample_size"],
            as_index=False,
        )
        .agg(recovery_accuracy=("family_correct", "mean"))
    )
    degrees = sorted(table["true_student_df"].unique())
    figure, axes = plt.subplots(
        1,
        len(degrees),
        figsize=(6.0 * len(degrees), 4.5),
        sharey=True,
    )
    axes = np.atleast_1d(axes)

    for axis, degrees_of_freedom in zip(axes, degrees):
        subset_df = table.loc[
            np.isclose(table["true_student_df"], degrees_of_freedom)
        ]
        for tau in sorted(subset_df["true_tau"].unique()):
            subset = subset_df.loc[
                np.isclose(subset_df["true_tau"], tau)
            ].sort_values("sample_size")
            axis.plot(
                subset["sample_size"],
                subset["recovery_accuracy"],
                marker="o",
                linewidth=2,
                label=rf"$\tau={tau:g}$",
            )
        axis.set_title(f"True Student t, df = {degrees_of_freedom:g}")
        axis.set_xlabel("Sample size")
        axis.set_xticks(sorted(table["sample_size"].unique()))
        axis.set_ylim(-0.02, 1.04)
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)

    axes[0].set_ylabel("Student-family recovery")
    figure.suptitle(
        "Student t is harder to distinguish as degrees of freedom increase",
        y=1.02,
    )
    figure.tight_layout()
    save_figure(figure, figures_directory, "student_recovery_by_df")


def plot_student_df_estimates(
    results: pd.DataFrame,
    *,
    figures_directory: Path,
) -> None:
    student = results.loc[
        (results["true_family"] == "student")
        & (results["selected_family"] == "student")
        & results["estimated_student_df"].notna()
    ].copy()

    degrees = sorted(student["true_student_df"].unique())
    figure, axes = plt.subplots(
        1,
        len(degrees),
        figsize=(6.0 * len(degrees), 4.8),
        sharey=False,
    )
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(20260710)

    for axis, degrees_of_freedom in zip(axes, degrees):
        subset_df = student.loc[
            np.isclose(student["true_student_df"], degrees_of_freedom)
        ]
        sample_sizes = sorted(subset_df["sample_size"].unique())
        positions = np.arange(len(sample_sizes), dtype=float)

        for position, sample_size in zip(positions, sample_sizes):
            values = subset_df.loc[
                subset_df["sample_size"] == sample_size,
                "estimated_student_df",
            ].to_numpy()
            jitter = rng.normal(0.0, 0.055, size=len(values))
            axis.scatter(
                np.full(len(values), position) + jitter,
                values,
                s=20,
                alpha=0.45,
                edgecolors="none",
            )
            if len(values):
                axis.plot(
                    [position - 0.18, position + 0.18],
                    [np.median(values), np.median(values)],
                    color="black",
                    linewidth=2.2,
                )

        axis.axhline(
            degrees_of_freedom,
            linestyle="--",
            linewidth=1.4,
            color="black",
            label="True df",
        )
        axis.set_xticks(positions, [str(value) for value in sample_sizes])
        axis.set_xlabel("Sample size")
        axis.set_ylabel("Estimated degrees of freedom")
        axis.set_title(
            f"True df = {degrees_of_freedom:g}\n"
            "(only runs selecting Student t)"
        )
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)

    figure.suptitle("Student-t degrees-of-freedom estimates", y=1.02)
    figure.tight_layout()
    save_figure(figure, figures_directory, "student_df_estimates")


def format_parameters(value: object) -> str:
    try:
        parameters = json.loads(str(value))
    except json.JSONDecodeError:
        return str(value)
    return "[" + ", ".join(f"{float(item):.3g}" for item in parameters) + "]"


def plot_representative_case(
    results: pd.DataFrame,
    case: RepresentativeCase,
    *,
    figures_directory: Path,
    grid_size: int,
    scatter_points: int,
) -> pd.Series:
    row = select_representative_run(results, case)

    true_model = reconstruct_model(
        row["true_family"],
        int(row["true_rotation"]),
        row["true_parameters"],
    )
    fitted_model = reconstruct_model(
        row["selected_family"],
        int(row["selected_rotation"]),
        row["estimated_parameters"],
    )

    u1, u2, true_density = density_grid(
        true_model,
        grid_size=grid_size,
    )
    _, _, fitted_density = density_grid(
        fitted_model,
        grid_size=grid_size,
    )
    difference = np.abs(true_density - fitted_density)

    combined_density = np.concatenate(
        [true_density.ravel(), fitted_density.ravel()]
    )
    density_limit = float(np.quantile(combined_density, 0.995))
    density_limit = max(density_limit, 1.0)
    difference_limit = float(np.quantile(difference, 0.995))
    difference_limit = max(difference_limit, 1e-8)

    true_sample = np.asarray(
        true_model.simulate(
            n=int(row["sample_size"]),
            qrng=False,
            seeds=[int(row["seed"])],
        ),
        dtype=float,
    )
    fitted_sample = np.asarray(
        fitted_model.simulate(
            n=int(row["sample_size"]),
            qrng=False,
            seeds=[int(row["seed"]) + 10_000_019],
        ),
        dtype=float,
    )

    number_points = min(scatter_points, len(true_sample))
    true_sample = true_sample[:number_points]
    fitted_sample = fitted_sample[:number_points]

    figure, axes = plt.subplots(2, 3, figsize=(13.2, 8.3))
    density_norm = Normalize(vmin=0.0, vmax=density_limit)

    true_contour = axes[0, 0].contourf(
        u1,
        u2,
        np.clip(true_density, 0.0, density_limit),
        levels=18,
        cmap="viridis",
        norm=density_norm,
    )
    axes[0, 0].set_title("True copula density")

    axes[0, 1].contourf(
        u1,
        u2,
        np.clip(fitted_density, 0.0, density_limit),
        levels=18,
        cmap="viridis",
        norm=density_norm,
    )
    axes[0, 1].set_title("Fitted copula density")

    difference_image = axes[0, 2].pcolormesh(
        u1,
        u2,
        np.clip(difference, 0.0, difference_limit),
        shading="auto",
        cmap="magma",
        vmin=0.0,
        vmax=difference_limit,
    )
    axes[0, 2].set_title("Absolute density difference")

    for axis in axes[0, :]:
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel(r"$u_1$")
        axis.set_ylabel(r"$u_2$")
        axis.set_aspect("equal")

    figure.colorbar(
        true_contour,
        ax=[axes[0, 0], axes[0, 1]],
        fraction=0.035,
        pad=0.02,
        label="Copula density (99.5% display cap)",
    )
    figure.colorbar(
        difference_image,
        ax=axes[0, 2],
        fraction=0.046,
        pad=0.04,
        label="Absolute difference (99.5% display cap)",
    )

    axes[1, 0].scatter(
        true_sample[:, 0],
        true_sample[:, 1],
        s=10,
        alpha=0.42,
        edgecolors="none",
    )
    axes[1, 0].set_title("Pseudo-observations from true copula")

    axes[1, 1].scatter(
        fitted_sample[:, 0],
        fitted_sample[:, 1],
        s=10,
        alpha=0.42,
        edgecolors="none",
    )
    axes[1, 1].set_title("Pseudo-observations from fitted copula")

    for axis in axes[1, :2]:
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel(r"$u_1$")
        axis.set_ylabel(r"$u_2$")
        axis.set_aspect("equal")

    axes[1, 2].axis("off")
    correctness = "yes" if bool(row["family_correct"]) else "no"
    metadata_lines = [
        f"Run: {row['run_id']}",
        "",
        f"True family: {family_label(row['true_family'])}",
        f"Selected family: {family_label(row['selected_family'])}",
        f"Exact family recovery: {correctness}",
        f"True / selected rotation: "
        f"{int(row['true_rotation'])}° / "
        f"{int(row['selected_rotation'])}°",
        "",
        f"True tau: {row['true_tau']:.3f}",
        f"Estimated tau: {row['estimated_tau']:.3f}",
        f"Absolute tau error: {row['tau_abs_error']:.4f}",
        "",
        f"True parameters: {format_parameters(row['true_parameters'])}",
        f"Fitted parameters: "
        f"{format_parameters(row['estimated_parameters'])}",
        f"n = {int(row['sample_size'])}, seed = {int(row['seed'])}",
    ]
    axes[1, 2].text(
        0.0,
        1.0,
        "\n".join(metadata_lines),
        ha="left",
        va="top",
        fontsize=9.2,
        linespacing=1.35,
        transform=axes[1, 2].transAxes,
    )

    title = (
        f"{family_label(row['true_family'])} ground truth: "
        f"tau = {row['true_tau']:.1f}, n = {int(row['sample_size'])}"
    )
    if row["true_family"] == "student":
        title += f", df = {row['true_student_df']:g}"
    figure.suptitle(title, fontsize=14, y=0.995)
    figure.subplots_adjust(
        left=0.06,
        right=0.96,
        bottom=0.07,
        top=0.92,
        wspace=0.32,
        hspace=0.32,
    )
    save_figure(
        figure,
        figures_directory,
        f"copula_case_{case.slug}",
    )
    return row


def markdown_table(
    rows: list[list[str]],
    headers: list[str],
) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_markdown_report(
    results: pd.DataFrame,
    family_summary: pd.DataFrame,
    representative_rows: list[pd.Series],
    *,
    output_directory: Path,
) -> None:
    statistics = summary_statistics(results)
    family_overall = (
        results.groupby("true_family", as_index=False)
        .agg(
            runs=("run_id", "size"),
            accuracy=("family_correct", "mean"),
            tau_mae=("tau_abs_error", "mean"),
        )
    )
    family_overall["_rank"] = family_overall["true_family"].map(
        {family: index for index, family in enumerate(FAMILY_ORDER)}
    )
    family_overall = family_overall.sort_values("_rank")

    family_rows = [
        [
            family_label(row.true_family),
            str(int(row.runs)),
            f"{100.0 * row.accuracy:.1f}%",
            f"{row.tau_mae:.4f}",
        ]
        for row in family_overall.itertuples()
    ]

    sample_rows = [
        [
            family_label(row.true_family),
            str(int(row.sample_size)),
            str(int(row.number_runs)),
            f"{100.0 * row.family_recovery_accuracy:.1f}%",
            f"{row.mean_tau_abs_error:.4f}",
            f"{row.mean_test_loglik_gap_from_oracle:.4f}",
        ]
        for row in family_summary.itertuples()
    ]

    representative_rows_markdown = [
        [
            str(row["run_id"]),
            family_label(str(row["true_family"])),
            family_label(str(row["selected_family"])),
            "yes" if bool(row["family_correct"]) else "no",
            f"{float(row['tau_abs_error']):.4f}",
        ]
        for row in representative_rows
    ]

    report = rf"""# Synthetic pair-copula recovery report

## Experiment coverage

The completed benchmark contains **{statistics['total_runs']} successful runs** over the six candidate families, three sample sizes, three nonzero dependence strengths, and two Student-t degrees-of-freedom settings. Models were selected by BIC from the configured candidate set.

## Headline findings

- Overall exact family recovery was **{100.0 * statistics['overall_family_accuracy']:.1f}%**.
- At \(n=1000\), exact family recovery was **{100.0 * statistics['accuracy_n1000']:.1f}%**.
- Mean absolute Kendall-tau error was **{statistics['overall_tau_mae']:.4f}** overall and **{statistics['tau_mae_n1000']:.4f}** at \(n=1000\).
- Student-t recovery was **{100.0 * statistics['student_df4_accuracy']:.1f}%** for true \(df=4\), but only **{100.0 * statistics['student_df10_accuracy']:.1f}%** for true \(df=10\). This is consistent with the Student copula approaching the Gaussian copula as degrees of freedom increase and with BIC penalizing the extra Student parameter.
- Across all Student scenarios, **{100.0 * statistics['student_to_gaussian_rate']:.1f}%** of runs were selected as Gaussian. Exact family identification is therefore harder than recovery of dependence strength.

## Recovery by family

{markdown_table(family_rows, ['True family', 'Runs', 'Exact recovery', 'Mean tau MAE'])}

## Recovery by family and sample size

{markdown_table(sample_rows, ['True family', 'n', 'Runs', 'Exact recovery', 'Mean tau MAE', 'Mean held-out loglik gap'])}

The held-out log-likelihood gap is the fitted mean test log-likelihood minus the oracle mean test log-likelihood. Values close to zero indicate that the selected model is predictively close to the known generating copula even when the exact family is not recovered.

## Representative copula figures

{markdown_table(representative_rows_markdown, ['Run', 'True family', 'Selected family', 'Exact', 'Tau absolute error'])}

Each representative figure contains true and fitted copula-density contours, an absolute density-difference heatmap, and pseudo-observations from both models. For visual stability, density displays are capped at their joint 99.5th percentile; model evaluation itself is not clipped.

## Interpretation

The experiment validates the bivariate calibration layer: dependence strength is recovered accurately, and the main one-parameter families are usually identified. The principal limitation is exact discrimination between Student-t and nearby symmetric alternatives, especially for \(df=10\). That limitation should be reported as a model-selection identifiability issue rather than as a complete failure of distributional recovery.

This benchmark does **not** yet validate empirical marginal estimation, two-parent conditional D-vines, vine-order selection, or full-DAG generation. Those remain separate synthetic-validation milestones.
"""

    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "README.md").write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    results_path = resolve_project_path(args.results)
    output_directory = resolve_project_path(args.output_directory)
    figures_directory = output_directory / "figures"
    tables_directory = output_directory / "tables"
    tables_directory.mkdir(parents=True, exist_ok=True)

    results = read_results(results_path)
    counts, proportions = confusion_tables(results)
    family_summary = family_sample_summary(results)
    scenario_table = scenario_summary(results)

    counts.to_csv(tables_directory / "confusion_matrix_counts.csv")
    proportions.to_csv(
        tables_directory / "confusion_matrix_row_proportions.csv"
    )
    family_summary.to_csv(
        tables_directory / "family_recovery_by_sample_size.csv",
        index=False,
    )
    scenario_table.to_csv(
        tables_directory / "recovery_by_scenario.csv",
        index=False,
    )

    plot_confusion_matrix(
        counts,
        normalized=False,
        figures_directory=figures_directory,
    )
    plot_confusion_matrix(
        proportions,
        normalized=True,
        figures_directory=figures_directory,
    )
    plot_family_recovery_by_sample_size(
        family_summary,
        figures_directory=figures_directory,
    )
    plot_recovery_by_tau(
        scenario_table,
        figures_directory=figures_directory,
    )
    plot_tau_recovery(
        results,
        figures_directory=figures_directory,
    )
    plot_tau_error_by_sample_size(
        family_summary,
        figures_directory=figures_directory,
    )
    plot_student_recovery(
        results,
        figures_directory=figures_directory,
    )
    plot_student_df_estimates(
        results,
        figures_directory=figures_directory,
    )

    representative_rows: list[pd.Series] = []
    for case in DEFAULT_REPRESENTATIVE_CASES:
        representative_rows.append(
            plot_representative_case(
                results,
                case,
                figures_directory=figures_directory,
                grid_size=args.grid_size,
                scatter_points=args.scatter_points,
            )
        )

    representative_table = pd.DataFrame(representative_rows)
    representative_columns = [
        "run_id",
        "true_family",
        "selected_family",
        "family_correct",
        "true_tau",
        "estimated_tau",
        "tau_abs_error",
        "true_student_df",
        "estimated_student_df",
        "sample_size",
        "seed",
        "true_parameters",
        "estimated_parameters",
    ]
    representative_table[representative_columns].to_csv(
        tables_directory / "representative_runs.csv",
        index=False,
    )

    statistics = summary_statistics(results)
    (output_directory / "headline_statistics.json").write_text(
        json.dumps(statistics, indent=2),
        encoding="utf-8",
    )

    write_markdown_report(
        results,
        family_summary,
        representative_rows,
        output_directory=output_directory,
    )

    print("Pair-copula recovery report complete.")
    print(f"Results: {results_path}")
    print(f"Report:  {output_directory / 'README.md'}")
    print(f"Figures: {figures_directory}")
    print(f"Tables:  {tables_directory}")


if __name__ == "__main__":
    main()
