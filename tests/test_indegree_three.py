"""Integration tests for maximum-indegree-three calibration and generation."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyvinecopulib as pv
import pytest

from copula_causal_sim.copulas.conditional_vine_indegree3 import (
    build_conditional_vine_library_indegree3,
    conditional_loglik_values_dvine4,
    fit_explicit_dvine4_order,
    fit_one_conditional_vine_indegree3,
    joint_loglik_values_dvine4,
    parent_loglik_values_dvine4,
    sample_child_from_fitted_dvine4,
)
from copula_causal_sim.copulas.conditional_vine_library import (
    build_bicop_fit_controls,
)
from copula_causal_sim.copulas.marginals import EmpiricalMarginalLibrary
from copula_causal_sim.generators.indegree_three import IndegreeThreeGenerator
from copula_causal_sim.synthetic.pair_copula import make_pair_copula_from_tau


COLUMNS = ["P1", "P2", "P3", "X"]


def _true_vine() -> pv.Vinecop:
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
        [
            make_pair_copula_from_tau("gumbel", 0.55),
        ],
    ]
    return pv.Vinecop.from_structure(
        structure=pv.DVineStructure([1, 2, 3, 4]),
        pair_copulas=pair_copulas,
    )


def _sample(n: int = 800, seed: int = 13) -> pd.DataFrame:
    data = np.asarray(
        _true_vine().simulate(n=n, qrng=False, seeds=[seed]),
        dtype=float,
    )
    return pd.DataFrame(data, columns=COLUMNS)


@pytest.fixture(scope="module")
def controls():
    result, _ = build_bicop_fit_controls(
        selection_criterion="bic",
        allow_rotations=True,
        num_threads=1,
    )
    return result


def test_four_dimensional_loglik_factorization(controls) -> None:
    data = _sample(n=600, seed=2)
    fitted = fit_explicit_dvine4_order(
        u_df=data,
        child="X",
        first_parent="P1",
        second_parent="P2",
        third_parent="P3",
        controls=controls,
    )

    joint = joint_loglik_values_dvine4(fitted, data)
    parent = parent_loglik_values_dvine4(fitted, data)
    conditional = conditional_loglik_values_dvine4(fitted, data)

    np.testing.assert_allclose(joint - parent, conditional, atol=1e-12)
    assert np.isfinite(joint).all()
    assert fitted.npars >= fitted.conditional_npars


def test_six_orders_are_scored_and_best_is_selected(
    controls,
    tmp_path: Path,
) -> None:
    data = _sample(n=900, seed=7)
    record = fit_one_conditional_vine_indegree3(
        u_df=data,
        child="X",
        parents=["P3", "P1", "P2"],
        dataset_id="synthetic",
        controls=controls,
        candidate_families=[
            "indep", "gaussian", "student", "clayton", "gumbel", "frank"
        ],
        selection_criterion="bic",
        model_dir=tmp_path,
        order_strategy="heldout_conditional_loglik",
        order_validation_fraction=0.20,
        order_selection_seed=41,
        minimum_validation_rows=50,
    )

    assert record.status == "ok", record.error
    assert len(record.candidate_order_scores) == 6
    successful_scores = [
        score for score in record.candidate_order_scores.values()
        if score is not None
    ]
    assert len(successful_scores) == 6
    assert record.selected_order[-1] == "X"
    assert set(record.selected_order[:-1]) == {"P1", "P2", "P3"}
    assert record.selected_order_score == pytest.approx(max(successful_scores))
    assert record.order_score_margin is not None
    assert record.order_score_margin >= -1e-12

    for field in (
        "bicop_parent_1_parent_2_file",
        "bicop_parent_2_parent_3_file",
        "bicop_parent_3_child_file",
        "bicop_parent_1_parent_3_given_parent_2_file",
        "bicop_parent_2_child_given_parent_3_file",
        "bicop_parent_1_child_given_parent_2_parent_3_file",
    ):
        assert Path(getattr(record, field)).exists()


def test_in_memory_conditional_sampling_is_reproducible(controls) -> None:
    data = _sample(n=700, seed=19)
    fitted = fit_explicit_dvine4_order(
        u_df=data,
        child="X",
        first_parent="P1",
        second_parent="P2",
        third_parent="P3",
        controls=controls,
    )
    parents = data[["P1", "P2", "P3"]].iloc[:150].reset_index(drop=True)
    first = sample_child_from_fitted_dvine4(fitted, parents, seed=123)
    second = sample_child_from_fitted_dvine4(fitted, parents, seed=123)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (150,)
    assert np.isfinite(first).all()
    assert np.all((first > 0.0) & (first < 1.0))


def test_library_builder_creates_one_record_for_unordered_parent_set(
    tmp_path: Path,
) -> None:
    data = _sample(n=500, seed=23)
    library = build_conditional_vine_library_indegree3(
        u_df=data,
        dataset_id="synthetic",
        model_dir=tmp_path / "models",
        order_strategy="heldout_conditional_loglik",
        order_validation_fraction=0.2,
        order_selection_seed=4,
        minimum_validation_rows=40,
        limit=1,
    )
    assert library.parent_set_size == 3
    assert len(library.records) == 1
    assert library.records[0].status == "ok"

    metadata_path = tmp_path / "library.pkl"
    library.save_pickle(metadata_path)
    loaded = library.load_pickle(metadata_path)
    assert loaded["parent_set_size"] == 3
    assert loaded["implementation"] == "explicit_4d_dvine_six_parent_orders_v1"


def test_full_indegree_three_generator_uses_saved_order(
    controls,
    tmp_path: Path,
) -> None:
    data = _sample(n=800, seed=29)
    record = fit_one_conditional_vine_indegree3(
        u_df=data,
        child="X",
        parents=["P1", "P2", "P3"],
        dataset_id="synthetic",
        controls=controls,
        candidate_families=[
            "indep", "gaussian", "student", "clayton", "gumbel", "frank"
        ],
        selection_criterion="bic",
        model_dir=tmp_path / "conditional_vines_indegree3",
        order_strategy="fixed",
    )
    assert record.status == "ok", record.error

    marginal_library = EmpiricalMarginalLibrary.fit(
        data,
        dataset_id="synthetic",
    )
    pair_metadata = {
        "dataset_id": "synthetic",
        "columns": COLUMNS,
        "records": [],
    }
    indegree2_metadata = {
        "dataset_id": "synthetic",
        "columns": COLUMNS,
        "parent_set_size": 2,
        "records": [],
    }
    indegree3_metadata = {
        "dataset_id": "synthetic",
        "columns": COLUMNS,
        "parent_set_size": 3,
        "records": [asdict(record)],
    }

    generator = IndegreeThreeGenerator(
        marginal_library=marginal_library,
        paircopula_metadata=pair_metadata,
        conditional_vine_metadata_indegree2=indegree2_metadata,
        conditional_vine_metadata_indegree3=indegree3_metadata,
        artifact_dir=tmp_path,
    )
    adjacency = np.zeros((4, 4), dtype=int)
    adjacency[0, 3] = 1
    adjacency[1, 3] = 1
    adjacency[2, 3] = 1

    first = generator.sample_u(adjacency, n_samples=250, seed=91)
    second = generator.sample_u(adjacency, n_samples=250, seed=91)
    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == COLUMNS
    assert np.isfinite(first.to_numpy()).all()
    assert np.all((first.to_numpy() > 0.0) & (first.to_numpy() < 1.0))
