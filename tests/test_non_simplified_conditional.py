from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

pv = pytest.importorskip("pyvinecopulib")

from copula_causal_sim.copulas.conditional_vine_library import (
    build_bicop_fit_controls,
    fit_one_conditional_vine,
)
from copula_causal_sim.copulas.non_simplified import (
    assign_quantile_bins,
    make_quantile_bin_edges,
)
from copula_causal_sim.generators.indegree_two import IndegreeTwoGenerator
from copula_causal_sim.synthetic.validation_suite import (
    simulate_non_simplified_conditional,
)


def test_quantile_bins_cover_training_and_boundary_values() -> None:
    conditioning = np.linspace(0.001, 0.999, 400)
    edges, assignments, counts = make_quantile_bin_edges(
        conditioning,
        n_bins=4,
        minimum_bin_size=90,
    )

    assert edges.shape == (5,)
    assert edges[0] == pytest.approx(0.0)
    assert edges[-1] == pytest.approx(1.0)
    assert np.all(np.diff(edges) > 0.0)
    assert sum(counts) == len(conditioning)
    assert min(counts) >= 90
    assert assignments.min() == 0
    assert assignments.max() == 3

    boundary_assignments = assign_quantile_bins(
        np.asarray([0.0, edges[1], edges[-2], 1.0]),
        edges,
    )
    assert boundary_assignments.tolist() == [0, 1, 3, 3]


def test_adaptive_fit_selects_multiple_bins_for_varying_dependence(
    tmp_path: Path,
) -> None:
    data = simulate_non_simplified_conditional(1600, seed=11).drop(
        columns=["true_tau"]
    )
    controls, families = build_bicop_fit_controls(
        selection_criterion="bic",
        allow_rotations=True,
        num_threads=1,
    )

    record = fit_one_conditional_vine(
        u_df=data,
        child="X",
        parent_1="P1",
        parent_2="P2",
        dataset_id="varying_dependence",
        controls=controls,
        candidate_families=families,
        selection_criterion="bic",
        model_dir=tmp_path,
        order_strategy="fixed",
        conditional_model_strategy="adaptive_quantile_bins",
        conditional_candidate_bin_counts=[1, 2, 3, 4],
        conditional_minimum_bin_size=120,
        conditional_validation_fraction=0.20,
        conditional_selection_seed=42,
        conditional_minimum_score_improvement=0.0,
    )

    assert record.status == "ok", record.error
    assert record.conditional_model_type == "quantile_binned"
    assert record.selected_number_bins > 1
    assert record.simplifying_assumption_selected is False
    assert len(record.conditional_bin_edges) == record.selected_number_bins + 1
    assert len(record.conditional_bin_model_files) == record.selected_number_bins
    assert sum(record.conditional_bin_counts) == len(data)
    assert record.selected_validation_score is not None
    assert record.simplified_validation_score is not None
    assert record.selected_validation_score >= record.simplified_validation_score


def test_simplified_strategy_preserves_single_conditional_model(
    tmp_path: Path,
) -> None:
    data = simulate_non_simplified_conditional(700, seed=23).drop(
        columns=["true_tau"]
    )
    controls, families = build_bicop_fit_controls()

    record = fit_one_conditional_vine(
        u_df=data,
        child="X",
        parent_1="P1",
        parent_2="P2",
        dataset_id="simplified",
        controls=controls,
        candidate_families=families,
        selection_criterion="bic",
        model_dir=tmp_path,
        order_strategy="fixed",
        conditional_model_strategy="simplified",
    )

    assert record.status == "ok", record.error
    assert record.conditional_model_type == "simplified"
    assert record.selected_number_bins == 1
    assert record.simplifying_assumption_selected is True
    assert record.conditional_bin_edges == [0.0, 1.0]
    assert record.conditional_bin_counts == [len(data)]
    assert record.conditional_bin_model_files == [
        record.bicop_parent_1_child_given_parent_2_file
    ]


def test_generator_loads_and_uses_bin_specific_models(tmp_path: Path) -> None:
    data = simulate_non_simplified_conditional(1400, seed=31).drop(
        columns=["true_tau"]
    )
    controls, families = build_bicop_fit_controls()
    record = fit_one_conditional_vine(
        u_df=data,
        child="X",
        parent_1="P1",
        parent_2="P2",
        dataset_id="generator_bins",
        controls=controls,
        candidate_families=families,
        selection_criterion="bic",
        model_dir=tmp_path / "conditional_vines_indegree2",
        order_strategy="fixed",
        conditional_model_strategy="adaptive_quantile_bins",
        conditional_candidate_bin_counts=[1, 2, 3, 4],
        conditional_minimum_bin_size=100,
        conditional_minimum_score_improvement=0.0,
    )
    assert record.status == "ok", record.error
    assert record.selected_number_bins > 1

    generator = IndegreeTwoGenerator.__new__(IndegreeTwoGenerator)
    generator.columns = ["P1", "P2", "X"]
    generator.column_to_index = {name: i for i, name in enumerate(generator.columns)}
    generator.artifact_dir = tmp_path
    generator.clip_eps = 1e-6
    generator.bicop_cache = {}
    payload = asdict(record)
    generator.conditional_lookup = {
        ("X", frozenset(["P1", "P2"])): payload
    }

    u_matrix = np.zeros((500, 3), dtype=float)
    rng_parents = np.random.default_rng(101)
    u_matrix[:, 0] = rng_parents.uniform(1e-6, 1.0 - 1e-6, 500)
    u_matrix[:, 1] = rng_parents.uniform(1e-6, 1.0 - 1e-6, 500)

    first = generator._sample_child_given_two_parents(
        child_name="X",
        parent_names=["P1", "P2"],
        u_matrix=u_matrix,
        rng=np.random.default_rng(777),
    )
    second = generator._sample_child_given_two_parents(
        child_name="X",
        parent_names=["P2", "P1"],
        u_matrix=u_matrix,
        rng=np.random.default_rng(777),
    )

    assert first.shape == (500,)
    assert np.all(np.isfinite(first))
    assert np.all((first > 0.0) & (first < 1.0))
    np.testing.assert_array_equal(first, second)
