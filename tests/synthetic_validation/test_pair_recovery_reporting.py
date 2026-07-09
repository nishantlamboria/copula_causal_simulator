"""Tests for synthetic pair-copula reporting utilities."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from copula_causal_sim.synthetic.pair_recovery_reporting import (
    RepresentativeCase,
    confusion_tables,
    density_grid,
    family_sample_summary,
    parse_parameter_vector,
    read_results,
    reconstruct_model,
    select_representative_run,
)


def make_results() -> pd.DataFrame:
    rows = [
        {
            "run_id": "gaussian_0",
            "scenario_id": "family_gaussian__tau_0p5",
            "true_family": "gaussian",
            "selected_family": "gaussian",
            "family_correct": True,
            "true_rotation": 0,
            "selected_rotation": 0,
            "true_tau": 0.5,
            "estimated_tau": 0.49,
            "tau_abs_error": 0.01,
            "true_student_df": np.nan,
            "estimated_student_df": np.nan,
            "student_df_abs_error": np.nan,
            "true_parameters": json.dumps([np.sqrt(0.5)]),
            "estimated_parameters": json.dumps([0.69]),
            "sample_size": 1000,
            "seed": 0,
            "test_loglik_gap_from_oracle": -0.001,
            "status": "success",
        },
        {
            "run_id": "gaussian_1",
            "scenario_id": "family_gaussian__tau_0p5",
            "true_family": "gaussian",
            "selected_family": "frank",
            "family_correct": False,
            "true_rotation": 0,
            "selected_rotation": 0,
            "true_tau": 0.5,
            "estimated_tau": 0.47,
            "tau_abs_error": 0.03,
            "true_student_df": np.nan,
            "estimated_student_df": np.nan,
            "student_df_abs_error": np.nan,
            "true_parameters": json.dumps([np.sqrt(0.5)]),
            "estimated_parameters": json.dumps([5.7]),
            "sample_size": 1000,
            "seed": 1,
            "test_loglik_gap_from_oracle": -0.01,
            "status": "success",
        },
        {
            "run_id": "student_0",
            "scenario_id": "family_student__tau_0p5__df_10",
            "true_family": "student",
            "selected_family": "gaussian",
            "family_correct": False,
            "true_rotation": 0,
            "selected_rotation": 0,
            "true_tau": 0.5,
            "estimated_tau": 0.505,
            "tau_abs_error": 0.005,
            "true_student_df": 10.0,
            "estimated_student_df": np.nan,
            "student_df_abs_error": np.nan,
            "true_parameters": json.dumps([np.sqrt(0.5), 10.0]),
            "estimated_parameters": json.dumps([0.71]),
            "sample_size": 1000,
            "seed": 2,
            "test_loglik_gap_from_oracle": -0.003,
            "status": "success",
        },
    ]
    return pd.DataFrame(rows)


def test_parse_parameter_vector() -> None:
    np.testing.assert_allclose(
        parse_parameter_vector("[0.7, 4.0]"),
        np.array([0.7, 4.0]),
    )
    assert parse_parameter_vector("[]").size == 0


def test_reconstruct_gaussian_model_and_density_grid() -> None:
    correlation = np.sin(np.pi * 0.5 / 2.0)
    model = reconstruct_model(
        family="gaussian",
        rotation=0,
        parameters=json.dumps([correlation]),
    )

    assert float(model.tau) == pytest.approx(0.5, abs=1e-8)

    u1, u2, density = density_grid(model, grid_size=25)
    assert u1.shape == (25, 25)
    assert u2.shape == (25, 25)
    assert density.shape == (25, 25)
    assert np.all(np.isfinite(density))
    assert np.all(density > 0.0)


def test_confusion_tables_are_ordered_and_normalized() -> None:
    results = make_results()
    counts, proportions = confusion_tables(results)

    assert counts.loc["gaussian", "gaussian"] == 1
    assert counts.loc["gaussian", "frank"] == 1
    assert counts.loc["student", "gaussian"] == 1
    assert proportions.loc["gaussian"].sum() == pytest.approx(1.0)
    assert proportions.loc["student"].sum() == pytest.approx(1.0)
    assert list(counts.index)[:2] == ["gaussian", "student"]


def test_family_sample_summary() -> None:
    summary = family_sample_summary(make_results())
    gaussian = summary.loc[
        (summary["true_family"] == "gaussian")
        & (summary["sample_size"] == 1000)
    ].iloc[0]

    assert int(gaussian["number_runs"]) == 2
    assert gaussian["family_recovery_accuracy"] == pytest.approx(0.5)
    assert gaussian["mean_tau_abs_error"] == pytest.approx(0.02)


def test_select_representative_run_prefers_requested_failure() -> None:
    results = make_results()
    case = RepresentativeCase(
        slug="student_failure",
        family="student",
        tau=0.5,
        sample_size=1000,
        student_df=10.0,
        prefer_correct=False,
        preferred_selected_family="gaussian",
    )

    selected = select_representative_run(results, case)
    assert selected["run_id"] == "student_0"
    assert selected["selected_family"] == "gaussian"


def test_read_results_normalizes_boolean_strings(tmp_path) -> None:
    results = make_results()
    results["family_correct"] = results["family_correct"].map(
        {True: "True", False: "False"}
    )
    path = tmp_path / "results.csv"
    results.to_csv(path, index=False)

    loaded = read_results(path)
    assert loaded["family_correct"].dtype == bool
    assert len(loaded) == 3
