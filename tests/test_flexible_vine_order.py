from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


pv = pytest.importorskip("pyvinecopulib")


from copula_causal_sim.copulas.conditional_vine_library import (
    build_bicop_fit_controls,
    fit_one_conditional_vine,
)


CLIP_EPS = 1e-6


def _copula_from_tau(family_name: str, tau: float):
    family = getattr(pv.BicopFamily, family_name)
    template = pv.Bicop.from_family(family=family)
    parameters = template.tau_to_parameters(float(tau))
    return pv.Bicop.from_family(
        family=family,
        parameters=parameters,
    )


def simulate_known_order_dvine(
    n_rows: int,
    seed: int,
) -> pd.DataFrame:
    """Simulate a simplified D-vine with generating order A--B--X."""
    c_ab = _copula_from_tau("clayton", 0.60)
    c_bx = _copula_from_tau("gumbel", 0.55)
    c_ax_given_b = _copula_from_tau("frank", 0.50)

    parent_sample = np.asarray(
        c_ab.simulate(n_rows, seeds=[seed]),
        dtype=float,
    )
    u_a = parent_sample[:, 0]
    u_b = parent_sample[:, 1]

    data_ab = np.asfortranarray(
        np.column_stack([u_a, u_b]),
        dtype=float,
    )
    w_a = np.asarray(c_ab.hfunc2(data_ab), dtype=float)
    w_a = np.clip(w_a, CLIP_EPS, 1.0 - CLIP_EPS)

    rng = np.random.default_rng(seed + 10_000)
    q = rng.uniform(CLIP_EPS, 1.0 - CLIP_EPS, size=n_rows)

    w_x = np.asarray(
        c_ax_given_b.hinv1(
            np.asfortranarray(np.column_stack([w_a, q]), dtype=float)
        ),
        dtype=float,
    )
    w_x = np.clip(w_x, CLIP_EPS, 1.0 - CLIP_EPS)

    u_x = np.asarray(
        c_bx.hinv1(
            np.asfortranarray(np.column_stack([u_b, w_x]), dtype=float)
        ),
        dtype=float,
    )
    u_x = np.clip(u_x, CLIP_EPS, 1.0 - CLIP_EPS)

    return pd.DataFrame({"A": u_a, "B": u_b, "X": u_x})


def test_known_generating_order_has_better_predictive_score(
    tmp_path: Path,
) -> None:
    """The selector recovers the better predictive representation.

    This does not assert causal-order identification. It tests selection of
    the D-vine order with superior held-out conditional likelihood under a
    known simplified generating construction.
    """
    controls, families = build_bicop_fit_controls(
        selection_criterion="bic",
        allow_rotations=True,
        num_threads=1,
    )

    selected_generating_order = 0
    number_repetitions = 3

    for seed in range(number_repetitions):
        data = simulate_known_order_dvine(
            n_rows=1400,
            seed=seed,
        )

        record = fit_one_conditional_vine(
            u_df=data,
            child="X",
            parent_1="A",
            parent_2="B",
            dataset_id="known_order",
            controls=controls,
            candidate_families=families,
            selection_criterion="bic",
            model_dir=tmp_path / f"seed_{seed}",
            clip_eps=CLIP_EPS,
            order_strategy="heldout_conditional_loglik",
            order_validation_fraction=0.20,
            order_selection_seed=42,
            minimum_validation_rows=50,
        )

        assert record.status == "ok"
        assert len(record.candidate_order_scores) == 2

        if record.selected_order == ["A", "B", "X"]:
            selected_generating_order += 1

    assert selected_generating_order >= 2
