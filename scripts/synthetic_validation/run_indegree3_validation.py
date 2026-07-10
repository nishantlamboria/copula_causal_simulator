"""Run a compact synthetic validation for a three-parent local D-vine.

The experiment generates a known four-dimensional D-vine, fits all six parent
orders with the production calibrator, and evaluates conditional generation on
held-out parent values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvinecopulib as pv
from scipy.spatial.distance import cdist
from scipy.stats import kendalltau


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from copula_causal_sim.copulas.conditional_vine_indegree3 import (  # noqa: E402
    fit_explicit_dvine4_order,
    fit_one_conditional_vine_indegree3,
    sample_child_from_fitted_dvine4,
)
from copula_causal_sim.copulas.conditional_vine_library import (  # noqa: E402
    build_bicop_fit_controls,
)
from copula_causal_sim.synthetic.pair_copula import (  # noqa: E402
    make_pair_copula_from_tau,
)


COLUMNS = ["P1", "P2", "P3", "X"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="reports/synthetic_validation/indegree_three_compact",
    )
    return parser.parse_args()


def make_true_vine() -> pv.Vinecop:
    pair_copulas = [
        [
            make_pair_copula_from_tau("clayton", 0.45),
            make_pair_copula_from_tau("gaussian", 0.35),
            make_pair_copula_from_tau("gumbel", 0.50),
        ],
        [
            make_pair_copula_from_tau("frank", 0.30),
            make_pair_copula_from_tau("clayton", 0.40),
        ],
        [make_pair_copula_from_tau("gumbel", 0.55)],
    ]
    return pv.Vinecop.from_structure(
        structure=pv.DVineStructure([1, 2, 3, 4]),
        pair_copulas=pair_copulas,
    )


def kendall_matrix(values: np.ndarray) -> np.ndarray:
    d = values.shape[1]
    matrix = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            tau = float(kendalltau(values[:, i], values[:, j]).statistic)
            matrix[i, j] = matrix[j, i] = tau
    return matrix


def off_diagonal_mae(first: np.ndarray, second: np.ndarray) -> float:
    mask = ~np.eye(first.shape[0], dtype=bool)
    return float(np.mean(np.abs(first[mask] - second[mask])))


def energy_distance(first: np.ndarray, second: np.ndarray) -> float:
    cross = cdist(first, second).mean()
    within_first = cdist(first, first).mean()
    within_second = cdist(second, second).mean()
    return float(2.0 * cross - within_first - within_second)


def main() -> None:
    args = parse_args()
    if args.n < 400:
        raise ValueError("Use at least 400 observations for this validation.")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    model_dir = output_dir / "models"
    for directory in (figure_dir, table_dir, model_dir):
        directory.mkdir(parents=True, exist_ok=True)

    true_vine = make_true_vine()
    values = np.asarray(
        true_vine.simulate(n=args.n, qrng=False, seeds=[args.seed]),
        dtype=float,
    )
    data = pd.DataFrame(values, columns=COLUMNS)

    controls, family_names = build_bicop_fit_controls(
        selection_criterion="bic",
        allow_rotations=True,
        num_threads=1,
    )
    record = fit_one_conditional_vine_indegree3(
        u_df=data,
        child="X",
        parents=["P1", "P2", "P3"],
        dataset_id="known_dvine4",
        controls=controls,
        candidate_families=family_names,
        selection_criterion="bic",
        model_dir=model_dir,
        order_strategy="heldout_conditional_loglik",
        order_validation_fraction=0.20,
        order_selection_seed=args.seed,
        minimum_validation_rows=75,
    )
    if record.status != "ok":
        raise RuntimeError(record.error)

    selected_parents = record.selected_order[:-1]
    fitted = fit_explicit_dvine4_order(
        u_df=data,
        child="X",
        first_parent=selected_parents[0],
        second_parent=selected_parents[1],
        third_parent=selected_parents[2],
        controls=controls,
    )

    rng = np.random.default_rng(args.seed + 100_003)
    test_indices = rng.choice(len(data), size=min(500, len(data) // 3), replace=False)
    test = data.iloc[test_indices].reset_index(drop=True)
    generated_child = sample_child_from_fitted_dvine4(
        fitted,
        test[selected_parents],
        seed=args.seed + 200_003,
    )
    generated = test[COLUMNS].copy()
    generated["X"] = generated_child

    real_matrix = kendall_matrix(test[COLUMNS].to_numpy())
    generated_matrix = kendall_matrix(generated[COLUMNS].to_numpy())

    candidate_table = pd.DataFrame(
        [
            {
                "order": key,
                "validation_mean_conditional_loglik": value,
                "status": record.candidate_order_statuses.get(key),
                "error": record.candidate_order_errors.get(key),
            }
            for key, value in record.candidate_order_scores.items()
        ]
    ).sort_values("validation_mean_conditional_loglik", ascending=False)
    candidate_table.to_csv(table_dir / "candidate_order_scores.csv", index=False)

    summary = {
        "n": int(args.n),
        "seed": int(args.seed),
        "generating_order": ["P1", "P2", "P3", "X"],
        "selected_order": record.selected_order,
        "all_six_orders_scored": int(candidate_table["validation_mean_conditional_loglik"].notna().sum()) == 6,
        "selected_order_score": record.selected_order_score,
        "second_best_order_score": record.second_best_order_score,
        "order_score_margin": record.order_score_margin,
        "heldout_pairwise_kendall_mae": off_diagonal_mae(real_matrix, generated_matrix),
        "heldout_energy_distance_4d": energy_distance(
            test[COLUMNS].to_numpy(), generated[COLUMNS].to_numpy()
        ),
        "generated_values_finite": bool(np.isfinite(generated.to_numpy()).all()),
        "generated_values_in_unit_interval": bool(
            np.all((generated.to_numpy() > 0.0) & (generated.to_numpy() < 1.0))
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    plot_table = candidate_table.dropna(
        subset=["validation_mean_conditional_loglik"]
    ).sort_values("validation_mean_conditional_loglik")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(
        plot_table["order"],
        plot_table["validation_mean_conditional_loglik"],
    )
    ax.set_xlabel("Mean held-out conditional log-likelihood")
    ax.set_ylabel("Candidate parent order")
    ax.set_title("Indegree-three D-vine order selection")
    fig.tight_layout()
    fig.savefig(figure_dir / "candidate_order_scores.png", dpi=220)
    fig.savefig(figure_dir / "candidate_order_scores.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.scatter(test["X"], generated["X"], alpha=0.45, s=18)
    ax.set_xlabel("Observed held-out child pseudo-observation")
    ax.set_ylabel("Conditionally generated child pseudo-observation")
    ax.set_title("Conditional generation at held-out parent values")
    fig.tight_layout()
    fig.savefig(figure_dir / "conditional_child_scatter.png", dpi=220)
    fig.savefig(figure_dir / "conditional_child_scatter.pdf")
    plt.close(fig)

    readme = f"""# Compact indegree-three synthetic validation

A known four-dimensional D-vine was simulated with generating order
`P1 -- P2 -- P3 -- X`. The production calibrator evaluated all six parent
permutations and selected `{ '--'.join(record.selected_order) }` using mean
held-out conditional log-likelihood.

- All six orders scored: **{summary['all_six_orders_scored']}**
- Score margin over second-best order: **{summary['order_score_margin']:.6f}**
- Held-out pairwise Kendall MAE: **{summary['heldout_pairwise_kendall_mae']:.6f}**
- Held-out four-dimensional energy distance: **{summary['heldout_energy_distance_4d']:.6f}**
- Generated values finite and inside `(0,1)`: **{summary['generated_values_finite'] and summary['generated_values_in_unit_interval']}**

This validation checks order scoring, the complete six-copula local mechanism,
and conditional generation for a node with three parents. It does not extend
the adaptive non-simplified binning method beyond indegree two.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Report written to: {output_dir}")


if __name__ == "__main__":
    main()
