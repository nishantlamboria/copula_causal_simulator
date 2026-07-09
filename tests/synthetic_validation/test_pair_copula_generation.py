"""Tests for synthetic pair-copula ground-truth generation."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import kendalltau

from copula_causal_sim.synthetic.pair_copula import (
    make_pair_copula_from_tau,
    normalize_family_name,
    simulate_pair_copula,
)


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("independence", "indep"),
        ("normal", "gaussian"),
        ("student_t", "student"),
        ("Clayton", "clayton"),
        ("GUMBEL", "gumbel"),
        ("frank", "frank"),
    ],
)
def test_normalize_family_name(
    alias: str,
    expected: str,
) -> None:
    assert normalize_family_name(alias) == expected


@pytest.mark.parametrize(
    "family",
    [
        "gaussian",
        "clayton",
        "gumbel",
        "frank",
    ],
)
def test_make_one_parameter_copula_has_requested_tau(
    family: str,
) -> None:
    model = make_pair_copula_from_tau(
        family=family,
        tau=0.5,
    )

    assert float(model.tau) == pytest.approx(
        0.5,
        abs=1e-8,
    )


def test_make_student_copula_has_requested_tau_and_df() -> None:
    model = make_pair_copula_from_tau(
        family="student",
        tau=0.5,
        student_df=6.0,
    )

    parameters = np.asarray(
        model.parameters,
        dtype=float,
    ).reshape(-1)

    assert float(model.tau) == pytest.approx(
        0.5,
        abs=1e-8,
    )
    assert parameters.shape == (2,)
    assert parameters[1] == pytest.approx(
        6.0,
        abs=1e-12,
    )


def test_independence_requires_zero_tau() -> None:
    model = make_pair_copula_from_tau(
        family="indep",
        tau=0.0,
    )

    assert float(model.tau) == pytest.approx(
        0.0,
        abs=1e-12,
    )

    with pytest.raises(
        ValueError,
        match="requires tau = 0",
    ):
        make_pair_copula_from_tau(
            family="indep",
            tau=0.2,
        )


@pytest.mark.parametrize(
    "family",
    [
        "gaussian",
        "clayton",
        "gumbel",
        "frank",
    ],
)
def test_simulated_sample_has_expected_shape_and_range(
    family: str,
) -> None:
    result = simulate_pair_copula(
        family=family,
        tau=0.5,
        n=250,
        seed=11,
    )

    assert result.data.shape == (250, 2)
    assert np.all(np.isfinite(result.data))
    assert np.all(result.data > 0.0)
    assert np.all(result.data < 1.0)


@pytest.mark.parametrize(
    "family",
    [
        "gaussian",
        "clayton",
        "gumbel",
        "frank",
    ],
)
def test_simulation_is_reproducible(
    family: str,
) -> None:
    first = simulate_pair_copula(
        family=family,
        tau=0.5,
        n=200,
        seed=42,
    )

    second = simulate_pair_copula(
        family=family,
        tau=0.5,
        n=200,
        seed=42,
    )

    np.testing.assert_array_equal(
        first.data,
        second.data,
    )


def test_different_seeds_produce_different_samples() -> None:
    first = simulate_pair_copula(
        family="gaussian",
        tau=0.5,
        n=200,
        seed=1,
    )

    second = simulate_pair_copula(
        family="gaussian",
        tau=0.5,
        n=200,
        seed=2,
    )

    assert not np.array_equal(
        first.data,
        second.data,
    )


@pytest.mark.parametrize(
    "family",
    [
        "gaussian",
        "clayton",
        "gumbel",
        "frank",
    ],
)
def test_empirical_tau_is_close_to_ground_truth(
    family: str,
) -> None:
    result = simulate_pair_copula(
        family=family,
        tau=0.5,
        n=4000,
        seed=123,
    )

    # scipy.stats.kendalltau returns an object with attribute 'statistic'.
    # Use getattr to appease static analyzers that may not recognize the attribute.
    kt_res = kendalltau(
        result.data[:, 0],
        result.data[:, 1],
    )
    empirical_tau = float(getattr(kt_res, "statistic"))

    assert empirical_tau == pytest.approx(
        0.5,
        abs=0.05,
    )


@pytest.mark.parametrize(
    ("argument", "value", "expected_exception"),
    [
        ("n", 1, ValueError),
        ("n", 20.5, TypeError),
        ("seed", -1, ValueError),
        ("seed", 2.5, TypeError),
    ],
)
def test_invalid_simulation_arguments(
    argument: str,
    value: object,
    expected_exception: type[Exception],
) -> None:
    arguments = {
        "family": "gaussian",
        "tau": 0.5,
        "n": 100,
        "seed": 1,
    }
    arguments[argument] = value

    with pytest.raises(expected_exception):
        simulate_pair_copula(**arguments)
