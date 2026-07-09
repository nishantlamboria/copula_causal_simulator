from __future__ import annotations

import numpy as np
import pandas as pd

from copula_causal_sim.copulas.marginals import make_pseudo_observations
from copula_causal_sim.synthetic.validation_suite import (
    PairSpec,
    build_fit_controls,
    fit_binned_conditional_copula,
    fit_simplified_dvine3,
    inverse_known_marginal,
    model_family_name,
    select_parent_order,
    simulate_non_simplified_conditional,
    simulate_simplified_dvine3,
    varying_tau,
)


def test_known_monotone_marginals_preserve_empirical_ranks() -> None:
    rng = np.random.default_rng(7)
    u = pd.DataFrame({"A": rng.uniform(size=200), "B": rng.uniform(size=200)})
    x = pd.DataFrame(
        {
            "A": inverse_known_marginal("gamma", u["A"].to_numpy()),
            "B": inverse_known_marginal("lognormal", u["B"].to_numpy()),
        }
    )
    expected = u.rank(method="average") / (len(u) + 1.0)
    observed = make_pseudo_observations(x)
    np.testing.assert_allclose(observed.to_numpy(), expected.to_numpy())


def test_simplified_dvine_generation_and_fit() -> None:
    data, _ = simulate_simplified_dvine3(
        n=1500,
        seed=3,
        first_middle=PairSpec("clayton", 0.35),
        middle_child=PairSpec("gumbel", 0.45),
        first_child_given_middle=PairSpec("frank", 0.50),
    )
    controls = build_fit_controls()
    fitted = fit_simplified_dvine3(data, "P1", "P2", "X", controls)
    assert model_family_name(fitted.c_first_middle) == "clayton"
    assert model_family_name(fitted.c_middle_child) == "gumbel"
    assert model_family_name(fitted.c_first_child_given_middle) == "frank"
    generated = fitted.sample(100, seed=5)
    assert list(generated.columns) == ["P1", "P2", "X"]
    assert np.all((generated.to_numpy() > 0.0) & (generated.to_numpy() < 1.0))


def test_parent_order_selection_returns_both_candidates() -> None:
    data, _ = simulate_simplified_dvine3(
        n=1000,
        seed=9,
        first_middle=PairSpec("clayton", 0.4),
        middle_child=PairSpec("gumbel", 0.5),
        first_child_given_middle=PairSpec("frank", 0.5),
    )
    controls = build_fit_controls()
    selected, candidates = select_parent_order(
        data.iloc[:700],
        data.iloc[700:850],
        ("P1", "P2"),
        "X",
        controls,
    )
    assert len(candidates) == 2
    assert set(candidates["order"]) == {"P1--P2--X", "P2--P1--X"}
    assert "--".join(selected.order) in set(candidates["order"])


def test_non_simplified_tau_varies_and_binned_model_fits() -> None:
    data = simulate_non_simplified_conditional(1000, seed=2)
    assert varying_tau(np.array([0.1]))[0] < varying_tau(np.array([0.9]))[0]
    binned = fit_binned_conditional_copula(
        data.iloc[:800],
        family="gaussian",
        n_bins=4,
        min_bin_size=100,
    )
    assert binned.n_bins == 4
    estimates = binned.tau_at(np.array([0.1, 0.9]))
    assert estimates[0] < estimates[1]
