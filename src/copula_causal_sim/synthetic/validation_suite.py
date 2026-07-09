"""End-to-end synthetic validation utilities for the copula causal simulator.

The module validates four layers of the project:

1. empirical marginal estimation followed by pair-copula recovery;
2. known three-variable simplified D-vines and data-driven parent order;
3. calibration and regeneration of a complete small DAG;
4. a controlled violation of the simplifying assumption, comparing a single
   conditional copula with a cross-validated quantile-binned alternative.

The routines are intentionally explicit.  They use the same pair-copula
families and h-function conventions as the production implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sin
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import pyvinecopulib as pv
from numpy.typing import NDArray
from scipy.spatial.distance import cdist, pdist
from scipy.special import ndtr, ndtri
from scipy.stats import beta as beta_distribution
from scipy.stats import gamma as gamma_distribution
from scipy.stats import lognorm, norm, t as student_distribution
from scipy.stats import kendalltau

from copula_causal_sim.copulas.marginals import (
    EmpiricalMarginalLibrary,
    make_pseudo_observations,
)
from copula_causal_sim.synthetic.pair_copula import (
    make_pair_copula_from_tau,
    normalize_family_name,
)

FloatArray = NDArray[np.float64]

EPS = 1e-6


# ---------------------------------------------------------------------------
# Generic pair-copula helpers
# ---------------------------------------------------------------------------


def clip_u(values: Any, eps: float = EPS) -> FloatArray:
    """Return finite pseudo-observations inside the open unit interval."""
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("Pseudo-observations contain non-finite values.")
    return np.clip(array, eps, 1.0 - eps)


def family_enum(family: str):
    """Resolve a canonical family name to the installed enum."""
    canonical = normalize_family_name(family)
    return getattr(pv.BicopFamily, canonical)


def model_family_name(model: pv.Bicop) -> str:
    """Return a stable lowercase family name for a fitted model."""
    family = model.family
    name = getattr(family, "name", None)
    if isinstance(name, str):
        return normalize_family_name(name)
    return normalize_family_name(str(family).split(".")[-1])


def model_parameters(model: pv.Bicop) -> list[float]:
    """Return a flat list of model parameters."""
    return np.asarray(model.parameters, dtype=float).reshape(-1).tolist()


def build_fit_controls(
    candidate_families: Sequence[str] = (
        "indep",
        "gaussian",
        "student",
        "clayton",
        "gumbel",
        "frank",
    ),
    selection_criterion: str = "bic",
    allow_rotations: bool = True,
) -> pv.FitControlsBicop:
    """Create pair-copula selection controls used throughout validation."""
    return pv.FitControlsBicop(
        family_set=[family_enum(name) for name in candidate_families],
        selection_criterion=str(selection_criterion).lower(),
        parametric_method="mle",
        allow_rotations=bool(allow_rotations),
        preselect_families=False,
        num_threads=1,
    )


def fit_bicop(
    data: FloatArray,
    controls: pv.FitControlsBicop,
) -> pv.Bicop:
    """Fit/select a bivariate copula on two pseudo-observation columns."""
    array = np.asfortranarray(clip_u(data))
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"Expected an n x 2 array, received {array.shape}.")
    return pv.Bicop.from_data(array, controls=controls, var_types=["c", "c"])


def fit_fixed_family_bicop(data: FloatArray, family: str) -> pv.Bicop:
    """Fit one specified family, retaining rotation selection where relevant."""
    controls = pv.FitControlsBicop(
        family_set=[family_enum(family)],
        selection_criterion="loglik",
        parametric_method="mle",
        allow_rotations=True,
        preselect_families=False,
        num_threads=1,
    )
    return fit_bicop(data, controls)


def log_pdf(model: pv.Bicop, data: FloatArray) -> FloatArray:
    """Evaluate pointwise log copula density with numerical protection."""
    density = np.asarray(
        model.pdf(np.asfortranarray(clip_u(data))),
        dtype=float,
    ).reshape(-1)
    return np.log(np.clip(density, 1e-300, None))


def sample_bicop_conditional_second(
    model: pv.Bicop,
    first: FloatArray,
    uniforms: FloatArray,
) -> FloatArray:
    """Sample/invert the second coordinate conditional on the first."""
    data = np.asfortranarray(
        np.column_stack([clip_u(first), clip_u(uniforms)])
    )
    return clip_u(model.hinv1(data))


def sample_bicop_conditional_first(
    model: pv.Bicop,
    second: FloatArray,
    uniforms: FloatArray,
) -> FloatArray:
    """Sample/invert the first coordinate conditional on the second."""
    data = np.asfortranarray(
        np.column_stack([clip_u(uniforms), clip_u(second)])
    )
    return clip_u(model.hinv2(data))


def deterministic_split(
    n: int,
    seed: int,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Return deterministic train, validation, and test indices."""
    if n < 10:
        raise ValueError("At least 10 observations are required.")
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("Split fractions must sum to one.")
    rng = np.random.default_rng(seed + 9_000_001)
    indices = rng.permutation(n)
    n_train = max(2, int(np.floor(fractions[0] * n)))
    n_valid = max(2, int(np.floor(fractions[1] * n)))
    if n_train + n_valid > n - 2:
        n_valid = n - n_train - 2
    return (
        indices[:n_train],
        indices[n_train : n_train + n_valid],
        indices[n_train + n_valid :],
    )


# ---------------------------------------------------------------------------
# Multivariate metrics
# ---------------------------------------------------------------------------


def pairwise_kendall_matrix(data: FloatArray) -> FloatArray:
    """Compute a symmetric Kendall-tau matrix."""
    array = np.asarray(data, dtype=float)
    d = array.shape[1]
    matrix = np.eye(d, dtype=float)
    for i in range(d):
        for j in range(i + 1, d):
            value = float(kendalltau(array[:, i], array[:, j]).statistic)
            matrix[i, j] = value
            matrix[j, i] = value
    return matrix


def off_diagonal_mae(first: FloatArray, second: FloatArray) -> float:
    """Mean absolute error over unique off-diagonal entries."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("Matrices must be square and have the same shape.")
    indices = np.triu_indices(first.shape[0], k=1)
    return float(np.mean(np.abs(first[indices] - second[indices])))


def multivariate_energy_distance(x: FloatArray, y: FloatArray) -> float:
    """Biased finite-sample multivariate energy distance."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    cross = float(cdist(x, y).mean())
    within_x = float(pdist(x).mean()) if len(x) > 1 else 0.0
    within_y = float(pdist(y).mean()) if len(y) > 1 else 0.0
    return max(0.0, 2.0 * cross - within_x - within_y)


def rbf_mmd2(x: FloatArray, y: FloatArray, max_points: int = 1000) -> float:
    """Biased squared RBF MMD with a pooled median-distance bandwidth."""
    x = np.asarray(x, dtype=float)[:max_points]
    y = np.asarray(y, dtype=float)[:max_points]
    pooled = np.vstack([x, y])
    distances = pdist(pooled, metric="sqeuclidean")
    positive = distances[distances > 0.0]
    bandwidth_sq = float(np.median(positive)) if positive.size else 1.0
    bandwidth_sq = max(bandwidth_sq, 1e-12)

    def kernel(a: FloatArray, b: FloatArray) -> FloatArray:
        return np.exp(-cdist(a, b, metric="sqeuclidean") / (2.0 * bandwidth_sq))

    return float(kernel(x, x).mean() + kernel(y, y).mean() - 2.0 * kernel(x, y).mean())


# ---------------------------------------------------------------------------
# Known marginal distributions
# ---------------------------------------------------------------------------


MARGINAL_NAMES = ("normal", "gamma", "lognormal", "student", "beta")


def inverse_known_marginal(name: str, u: FloatArray) -> FloatArray:
    """Transform uniforms using a fixed continuous marginal distribution."""
    u = clip_u(u)
    canonical = str(name).strip().lower()
    if canonical == "normal":
        return np.asarray(norm.ppf(u, loc=1.0, scale=2.0), dtype=float)
    if canonical == "gamma":
        return np.asarray(gamma_distribution.ppf(u, a=2.5, scale=1.3), dtype=float)
    if canonical == "lognormal":
        return np.asarray(lognorm.ppf(u, s=0.65, scale=np.exp(0.3)), dtype=float)
    if canonical == "student":
        return np.asarray(student_distribution.ppf(u, df=5.0, loc=-0.5, scale=1.4), dtype=float)
    if canonical == "beta":
        return np.asarray(beta_distribution.ppf(u, a=2.0, b=5.0), dtype=float)
    raise ValueError(f"Unknown marginal {name!r}; expected one of {MARGINAL_NAMES}.")


def normalized_quantile_error(
    real: FloatArray,
    generated: FloatArray,
    probabilities: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
) -> float:
    """Median absolute quantile error normalized by the real IQR."""
    real = np.asarray(real, dtype=float)
    generated = np.asarray(generated, dtype=float)
    real_q = np.quantile(real, probabilities)
    generated_q = np.quantile(generated, probabilities)
    scale = float(np.quantile(real, 0.75) - np.quantile(real, 0.25))
    if scale <= 1e-12:
        scale = max(float(np.std(real)), 1.0)
    return float(np.median(np.abs(real_q - generated_q) / scale))


# ---------------------------------------------------------------------------
# Simplified three-variable D-vines
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairSpec:
    family: str
    tau: float
    student_df: float = 4.0

    def model(self) -> pv.Bicop:
        return make_pair_copula_from_tau(
            self.family,
            self.tau,
            student_df=self.student_df,
        )


@dataclass
class FittedDvine3:
    """Explicit three-variable D-vine in order first -- middle -- child."""

    first_name: str
    middle_name: str
    child_name: str
    c_first_middle: pv.Bicop
    c_middle_child: pv.Bicop
    c_first_child_given_middle: pv.Bicop

    @property
    def order(self) -> tuple[str, str, str]:
        return self.first_name, self.middle_name, self.child_name

    def conditional_loglik(self, data: pd.DataFrame) -> FloatArray:
        first = clip_u(data[self.first_name].to_numpy())
        middle = clip_u(data[self.middle_name].to_numpy())
        child = clip_u(data[self.child_name].to_numpy())
        pair_fm = np.column_stack([first, middle])
        pair_mc = np.column_stack([middle, child])
        w_first = clip_u(self.c_first_middle.hfunc2(np.asfortranarray(pair_fm)))
        w_child = clip_u(self.c_middle_child.hfunc1(np.asfortranarray(pair_mc)))
        conditional_pair = np.column_stack([w_first, w_child])
        return log_pdf(self.c_middle_child, pair_mc) + log_pdf(
            self.c_first_child_given_middle,
            conditional_pair,
        )

    def joint_loglik(self, data: pd.DataFrame) -> FloatArray:
        first = clip_u(data[self.first_name].to_numpy())
        middle = clip_u(data[self.middle_name].to_numpy())
        pair_fm = np.column_stack([first, middle])
        return log_pdf(self.c_first_middle, pair_fm) + self.conditional_loglik(data)

    def sample_child_given_parents(
        self,
        data: pd.DataFrame,
        seed: int,
    ) -> FloatArray:
        """Sample the child for already generated parent values."""
        rng = np.random.default_rng(seed)
        first = clip_u(data[self.first_name].to_numpy())
        middle = clip_u(data[self.middle_name].to_numpy())
        w_first = clip_u(
            self.c_first_middle.hfunc2(
                np.asfortranarray(np.column_stack([first, middle]))
            )
        )
        w_child = sample_bicop_conditional_second(
            self.c_first_child_given_middle,
            w_first,
            rng.uniform(EPS, 1.0 - EPS, size=len(data)),
        )
        return sample_bicop_conditional_second(
            self.c_middle_child,
            middle,
            w_child,
        )

    def sample(self, n: int, seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        middle = rng.uniform(EPS, 1.0 - EPS, size=n)
        q_first = rng.uniform(EPS, 1.0 - EPS, size=n)
        first = sample_bicop_conditional_first(
            self.c_first_middle,
            middle,
            q_first,
        )
        w_first = clip_u(
            self.c_first_middle.hfunc2(
                np.asfortranarray(np.column_stack([first, middle]))
            )
        )
        q_child = rng.uniform(EPS, 1.0 - EPS, size=n)
        w_child = sample_bicop_conditional_second(
            self.c_first_child_given_middle,
            w_first,
            q_child,
        )
        child = sample_bicop_conditional_second(
            self.c_middle_child,
            middle,
            w_child,
        )
        return pd.DataFrame(
            {
                self.first_name: first,
                self.middle_name: middle,
                self.child_name: child,
            }
        )

    def component_summary(self) -> dict[str, Any]:
        return {
            "family_first_middle": model_family_name(self.c_first_middle),
            "family_middle_child": model_family_name(self.c_middle_child),
            "family_first_child_given_middle": model_family_name(
                self.c_first_child_given_middle
            ),
            "tau_first_middle": float(self.c_first_middle.tau),
            "tau_middle_child": float(self.c_middle_child.tau),
            "tau_first_child_given_middle": float(
                self.c_first_child_given_middle.tau
            ),
            "parameters_first_middle": model_parameters(self.c_first_middle),
            "parameters_middle_child": model_parameters(self.c_middle_child),
            "parameters_first_child_given_middle": model_parameters(
                self.c_first_child_given_middle
            ),
        }


def simulate_simplified_dvine3(
    n: int,
    seed: int,
    first_middle: PairSpec,
    middle_child: PairSpec,
    first_child_given_middle: PairSpec,
    names: tuple[str, str, str] = ("P1", "P2", "X"),
) -> tuple[pd.DataFrame, FittedDvine3]:
    """Generate a known simplified three-variable D-vine."""
    true_model = FittedDvine3(
        first_name=names[0],
        middle_name=names[1],
        child_name=names[2],
        c_first_middle=first_middle.model(),
        c_middle_child=middle_child.model(),
        c_first_child_given_middle=first_child_given_middle.model(),
    )
    return true_model.sample(n=n, seed=seed), true_model


def fit_simplified_dvine3(
    data: pd.DataFrame,
    first_name: str,
    middle_name: str,
    child_name: str,
    controls: pv.FitControlsBicop,
) -> FittedDvine3:
    """Fit an explicit D-vine in the specified order."""
    first = clip_u(data[first_name].to_numpy())
    middle = clip_u(data[middle_name].to_numpy())
    child = clip_u(data[child_name].to_numpy())

    pair_fm = np.asfortranarray(np.column_stack([first, middle]))
    pair_mc = np.asfortranarray(np.column_stack([middle, child]))
    c_fm = fit_bicop(pair_fm, controls)
    c_mc = fit_bicop(pair_mc, controls)
    w_first = clip_u(c_fm.hfunc2(pair_fm))
    w_child = clip_u(c_mc.hfunc1(pair_mc))
    c_fc_m = fit_bicop(np.column_stack([w_first, w_child]), controls)
    return FittedDvine3(
        first_name=first_name,
        middle_name=middle_name,
        child_name=child_name,
        c_first_middle=c_fm,
        c_middle_child=c_mc,
        c_first_child_given_middle=c_fc_m,
    )


def select_parent_order(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    parent_names: tuple[str, str],
    child_name: str,
    controls: pv.FitControlsBicop,
) -> tuple[FittedDvine3, pd.DataFrame]:
    """Select between both parent orders using validation conditional loglik."""
    candidates: list[tuple[FittedDvine3, dict[str, Any]]] = []
    for first, middle in (parent_names, parent_names[::-1]):
        model = fit_simplified_dvine3(
            train,
            first_name=first,
            middle_name=middle,
            child_name=child_name,
            controls=controls,
        )
        score = float(model.conditional_loglik(validation).mean())
        row = {
            "first_parent": first,
            "middle_parent": middle,
            "order": f"{first}--{middle}--{child_name}",
            "validation_conditional_loglik_mean": score,
            **model.component_summary(),
        }
        candidates.append((model, row))
    table = pd.DataFrame([row for _, row in candidates]).sort_values(
        "validation_conditional_loglik_mean",
        ascending=False,
    )
    selected_order = str(table.iloc[0]["order"])
    selected = next(
        model
        for model, row in candidates
        if row["order"] == selected_order
    )
    return selected, table.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Marginal robustness experiment
# ---------------------------------------------------------------------------


def run_marginal_validation(
    *,
    families: Sequence[str],
    taus: Sequence[float],
    marginal_pairs: Sequence[tuple[str, str]],
    sample_sizes: Sequence[int],
    seeds: Sequence[int],
    controls: pv.FitControlsBicop,
) -> pd.DataFrame:
    """Validate pair recovery after known marginals and empirical ranks."""
    rows: list[dict[str, Any]] = []
    for family in families:
        for tau in taus:
            for marginal_1, marginal_2 in marginal_pairs:
                for n in sample_sizes:
                    for seed in seeds:
                        true_model = make_pair_copula_from_tau(family, tau)
                        u = np.asarray(
                            true_model.simulate(n=n, qrng=False, seeds=[int(seed)]),
                            dtype=float,
                        )
                        x = pd.DataFrame(
                            {
                                "A": inverse_known_marginal(marginal_1, u[:, 0]),
                                "B": inverse_known_marginal(marginal_2, u[:, 1]),
                            }
                        )
                        empirical_library = EmpiricalMarginalLibrary.fit(
                            x,
                            dataset_id="synthetic_marginal",
                        )
                        u_hat_df = make_pseudo_observations(x)
                        fitted = fit_bicop(u_hat_df.to_numpy(), controls)
                        rng = np.random.default_rng(seed + 200_003)
                        generated_u = np.asarray(
                            fitted.simulate(
                                n=n,
                                qrng=False,
                                seeds=[int(rng.integers(0, 2**31 - 1))],
                            ),
                            dtype=float,
                        )
                        generated_x = empirical_library.inverse_transform(
                            pd.DataFrame(generated_u, columns=["A", "B"])
                        )
                        row = {
                            "true_family": normalize_family_name(family),
                            "selected_family": model_family_name(fitted),
                            "family_correct": model_family_name(fitted)
                            == normalize_family_name(family),
                            "true_tau": float(tau),
                            "estimated_tau": float(fitted.tau),
                            "tau_abs_error": abs(float(fitted.tau) - float(tau)),
                            "marginal_1": marginal_1,
                            "marginal_2": marginal_2,
                            "sample_size": int(n),
                            "seed": int(seed),
                            "marginal_error_A": normalized_quantile_error(
                                x["A"].to_numpy(), generated_x["A"].to_numpy()
                            ),
                            "marginal_error_B": normalized_quantile_error(
                                x["B"].to_numpy(), generated_x["B"].to_numpy()
                            ),
                            "max_rank_invariance_error": float(
                                np.max(
                                    np.abs(
                                        u_hat_df.to_numpy()
                                        - pd.DataFrame(u, columns=["A", "B"])
                                        .rank(method="average")
                                        .to_numpy()
                                        / (n + 1.0)
                                    )
                                )
                            ),
                        }
                        row["mean_marginal_error"] = float(
                            0.5 * (row["marginal_error_A"] + row["marginal_error_B"])
                        )
                        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# D-vine recovery and order-selection experiment
# ---------------------------------------------------------------------------


def run_dvine_validation(
    *,
    scenarios: Sequence[dict[str, Any]],
    sample_sizes: Sequence[int],
    seeds: Sequence[int],
    controls: pv.FitControlsBicop,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate known D-vine components, conditional sampling, and order selection."""
    rows: list[dict[str, Any]] = []
    candidate_rows: list[pd.DataFrame] = []
    for scenario in scenarios:
        scenario_id = str(scenario["id"])
        true_specs = {
            key: PairSpec(**scenario[key])
            for key in ("first_middle", "middle_child", "conditional")
        }
        for n in sample_sizes:
            for seed in seeds:
                data, true_model = simulate_simplified_dvine3(
                    n=n,
                    seed=seed,
                    first_middle=true_specs["first_middle"],
                    middle_child=true_specs["middle_child"],
                    first_child_given_middle=true_specs["conditional"],
                )
                train_idx, valid_idx, test_idx = deterministic_split(n, seed)
                train = data.iloc[train_idx].reset_index(drop=True)
                validation = data.iloc[valid_idx].reset_index(drop=True)
                test = data.iloc[test_idx].reset_index(drop=True)

                true_order_fit = fit_simplified_dvine3(
                    train,
                    first_name="P1",
                    middle_name="P2",
                    child_name="X",
                    controls=controls,
                )
                selected, candidate_table = select_parent_order(
                    train,
                    validation,
                    parent_names=("P1", "P2"),
                    child_name="X",
                    controls=controls,
                )
                candidate_table.insert(0, "scenario_id", scenario_id)
                candidate_table.insert(1, "sample_size", n)
                candidate_table.insert(2, "seed", seed)
                candidate_rows.append(candidate_table)

                generated = selected.sample(n=len(test), seed=seed + 700_001)
                generated = generated[["P1", "P2", "X"]]
                test_array = test[["P1", "P2", "X"]].to_numpy()
                generated_array = generated.to_numpy()
                selected_summary = selected.component_summary()
                true_order_summary = true_order_fit.component_summary()

                selected_test_ll = float(selected.conditional_loglik(test).mean())
                true_test_ll = float(true_model.conditional_loglik(test).mean())
                row = {
                    "scenario_id": scenario_id,
                    "sample_size": int(n),
                    "seed": int(seed),
                    "selected_order": "--".join(selected.order),
                    "generating_order": "P1--P2--X",
                    "generating_order_selected": selected.order == ("P1", "P2", "X"),
                    "selected_test_conditional_loglik_mean": selected_test_ll,
                    "oracle_test_conditional_loglik_mean": true_test_ll,
                    "test_loglik_gap_from_oracle": selected_test_ll - true_test_ll,
                    "pairwise_kendall_mae": off_diagonal_mae(
                        pairwise_kendall_matrix(test_array),
                        pairwise_kendall_matrix(generated_array),
                    ),
                    "energy_distance_3d": multivariate_energy_distance(
                        test_array,
                        generated_array,
                    ),
                    "mmd2_3d": rbf_mmd2(test_array, generated_array),
                    "true_family_first_middle": normalize_family_name(
                        true_specs["first_middle"].family
                    ),
                    "true_family_middle_child": normalize_family_name(
                        true_specs["middle_child"].family
                    ),
                    "true_family_conditional": normalize_family_name(
                        true_specs["conditional"].family
                    ),
                    "fitted_true_order_family_first_middle": true_order_summary[
                        "family_first_middle"
                    ],
                    "fitted_true_order_family_middle_child": true_order_summary[
                        "family_middle_child"
                    ],
                    "fitted_true_order_family_conditional": true_order_summary[
                        "family_first_child_given_middle"
                    ],
                    "true_order_first_middle_correct": true_order_summary[
                        "family_first_middle"
                    ]
                    == normalize_family_name(true_specs["first_middle"].family),
                    "true_order_middle_child_correct": true_order_summary[
                        "family_middle_child"
                    ]
                    == normalize_family_name(true_specs["middle_child"].family),
                    "true_order_conditional_correct": true_order_summary[
                        "family_first_child_given_middle"
                    ]
                    == normalize_family_name(true_specs["conditional"].family),
                    "selected_family_first_middle": selected_summary[
                        "family_first_middle"
                    ],
                    "selected_family_middle_child": selected_summary[
                        "family_middle_child"
                    ],
                    "selected_family_conditional": selected_summary[
                        "family_first_child_given_middle"
                    ],
                }
                rows.append(row)
    return pd.DataFrame(rows), pd.concat(candidate_rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Complete known-DAG experiment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KnownDagSpecification:
    """Five-node DAG with roots, one-parent nodes, and one two-parent node."""

    x1_x2: PairSpec
    x2_x3: PairSpec
    x1_x3_given_x2: PairSpec
    x3_x4: PairSpec
    x2_x5: PairSpec


def simulate_known_dag_u(
    n: int,
    seed: int,
    specification: KnownDagSpecification,
) -> tuple[pd.DataFrame, dict[str, pv.Bicop]]:
    """Generate U for X1,X2 -> X3 -> X4 and X2 -> X5."""
    rng = np.random.default_rng(seed)
    c12 = specification.x1_x2.model()
    c23 = specification.x2_x3.model()
    c13_2 = specification.x1_x3_given_x2.model()
    c34 = specification.x3_x4.model()
    c25 = specification.x2_x5.model()

    x2 = rng.uniform(EPS, 1.0 - EPS, n)
    x1 = sample_bicop_conditional_first(
        c12,
        x2,
        rng.uniform(EPS, 1.0 - EPS, n),
    )
    w1 = clip_u(
        c12.hfunc2(np.asfortranarray(np.column_stack([x1, x2])))
    )
    w3 = sample_bicop_conditional_second(
        c13_2,
        w1,
        rng.uniform(EPS, 1.0 - EPS, n),
    )
    x3 = sample_bicop_conditional_second(c23, x2, w3)
    x4 = sample_bicop_conditional_second(
        c34,
        x3,
        rng.uniform(EPS, 1.0 - EPS, n),
    )
    x5 = sample_bicop_conditional_second(
        c25,
        x2,
        rng.uniform(EPS, 1.0 - EPS, n),
    )
    return (
        pd.DataFrame({"X1": x1, "X2": x2, "X3": x3, "X4": x4, "X5": x5}),
        {"c12": c12, "c23": c23, "c13_2": c13_2, "c34": c34, "c25": c25},
    )


def transform_known_dag_marginals(u: pd.DataFrame) -> pd.DataFrame:
    """Apply heterogeneous continuous marginals to the five DAG variables."""
    names = {
        "X1": "normal",
        "X2": "gamma",
        "X3": "lognormal",
        "X4": "student",
        "X5": "beta",
    }
    return pd.DataFrame(
        {column: inverse_known_marginal(names[column], u[column].to_numpy()) for column in u.columns}
    )


def run_full_dag_validation(
    *,
    specification: KnownDagSpecification,
    sample_sizes: Sequence[int],
    seeds: Sequence[int],
    controls: pv.FitControlsBicop,
) -> pd.DataFrame:
    """Calibrate and regenerate a complete known five-node DAG."""
    rows: list[dict[str, Any]] = []
    for n in sample_sizes:
        for seed in seeds:
            true_u, _ = simulate_known_dag_u(n, seed, specification)
            true_x = transform_known_dag_marginals(true_u)
            marginal_library = EmpiricalMarginalLibrary.fit(
                true_x,
                dataset_id="known_dag",
            )
            u_hat = make_pseudo_observations(true_x)
            train_idx, valid_idx, test_idx = deterministic_split(n, seed)
            train = u_hat.iloc[train_idx].reset_index(drop=True)
            validation = u_hat.iloc[valid_idx].reset_index(drop=True)
            test = u_hat.iloc[test_idx].reset_index(drop=True)

            fitted_x3, order_table = select_parent_order(
                train[["X1", "X2", "X3"]],
                validation[["X1", "X2", "X3"]],
                parent_names=("X1", "X2"),
                child_name="X3",
                controls=controls,
            )
            fitted_x2 = fit_bicop(train[["X1", "X2"]].to_numpy(), controls)
            fitted_x4 = fit_bicop(train[["X3", "X4"]].to_numpy(), controls)
            fitted_x5 = fit_bicop(train[["X2", "X5"]].to_numpy(), controls)

            rng = np.random.default_rng(seed + 800_003)
            n_test = len(test)
            generated_u = pd.DataFrame(index=np.arange(n_test), columns=true_u.columns, dtype=float)
            # Exact topological generation for the known DAG:
            # X1 -> X2; X1,X2 -> X3; X3 -> X4; X2 -> X5.
            generated_u["X1"] = rng.uniform(EPS, 1.0 - EPS, n_test)
            generated_u["X2"] = sample_bicop_conditional_second(
                fitted_x2,
                generated_u["X1"].to_numpy(),
                rng.uniform(EPS, 1.0 - EPS, n_test),
            )
            generated_u["X3"] = fitted_x3.sample_child_given_parents(
                generated_u[["X1", "X2"]],
                seed=seed + 810_001,
            )
            generated_u["X4"] = sample_bicop_conditional_second(
                fitted_x4,
                generated_u["X3"].to_numpy(),
                rng.uniform(EPS, 1.0 - EPS, n_test),
            )
            generated_u["X5"] = sample_bicop_conditional_second(
                fitted_x5,
                generated_u["X2"].to_numpy(),
                rng.uniform(EPS, 1.0 - EPS, n_test),
            )
            generated_x = marginal_library.inverse_transform(generated_u)

            test_u = test[true_u.columns].to_numpy()
            generated_u_array = generated_u.to_numpy()
            test_x = true_x.iloc[test_idx].reset_index(drop=True)
            marginal_errors = [
                normalized_quantile_error(
                    test_x[column].to_numpy(),
                    generated_x[column].to_numpy(),
                )
                for column in true_x.columns
            ]
            selected_order = "--".join(fitted_x3.order)
            row = {
                "sample_size": int(n),
                "seed": int(seed),
                "selected_x3_order": selected_order,
                "generating_x3_order_selected": selected_order == "X1--X2--X3",
                "x3_order_validation_advantage": float(
                    order_table.iloc[0]["validation_conditional_loglik_mean"]
                    - order_table.iloc[1]["validation_conditional_loglik_mean"]
                ),
                "x2_true_family": normalize_family_name(specification.x1_x2.family),
                "x2_selected_family": model_family_name(fitted_x2),
                "x2_family_correct": model_family_name(fitted_x2)
                == normalize_family_name(specification.x1_x2.family),
                "x4_true_family": normalize_family_name(specification.x3_x4.family),
                "x4_selected_family": model_family_name(fitted_x4),
                "x4_family_correct": model_family_name(fitted_x4)
                == normalize_family_name(specification.x3_x4.family),
                "x5_true_family": normalize_family_name(specification.x2_x5.family),
                "x5_selected_family": model_family_name(fitted_x5),
                "x5_family_correct": model_family_name(fitted_x5)
                == normalize_family_name(specification.x2_x5.family),
                "pairwise_kendall_mae": off_diagonal_mae(
                    pairwise_kendall_matrix(test_u),
                    pairwise_kendall_matrix(generated_u_array),
                ),
                "energy_distance_5d": multivariate_energy_distance(
                    test_u,
                    generated_u_array,
                ),
                "mmd2_5d": rbf_mmd2(test_u, generated_u_array),
                "mean_marginal_quantile_error": float(np.mean(marginal_errors)),
                "max_marginal_quantile_error": float(np.max(marginal_errors)),
            }
            for column, value in zip(true_x.columns, marginal_errors):
                row[f"marginal_error_{column}"] = float(value)
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Non-simplified conditional copula experiment
# ---------------------------------------------------------------------------


def varying_tau(u2: FloatArray, low: float = 0.1, high: float = 0.75) -> FloatArray:
    """Smooth monotone conditional Kendall-tau function used as ground truth."""
    u2 = np.asarray(u2, dtype=float)
    return low + (high - low) * u2


def gaussian_copula_log_density(
    first: FloatArray,
    second: FloatArray,
    rho: FloatArray,
) -> FloatArray:
    """Pointwise Gaussian copula log density for observation-specific rho."""
    z1 = ndtri(clip_u(first))
    z2 = ndtri(clip_u(second))
    rho = np.clip(np.asarray(rho, dtype=float), -0.995, 0.995)
    one_minus = 1.0 - rho**2
    return (
        -0.5 * np.log(one_minus)
        - (rho**2 * (z1**2 + z2**2) - 2.0 * rho * z1 * z2)
        / (2.0 * one_minus)
    )


def simulate_non_simplified_conditional(
    n: int,
    seed: int,
) -> pd.DataFrame:
    """Generate P1,P2,X where C(P1,X|P2=u2) has varying Gaussian tau."""
    rng = np.random.default_rng(seed)
    p2 = rng.uniform(EPS, 1.0 - EPS, n)
    p1 = rng.uniform(EPS, 1.0 - EPS, n)
    tau = varying_tau(p2)
    rho = np.sin(0.5 * pi * tau)
    z1 = ndtri(p1)
    innovation = rng.normal(size=n)
    zx = rho * z1 + np.sqrt(1.0 - rho**2) * innovation
    x = clip_u(ndtr(zx))
    return pd.DataFrame({"P1": p1, "P2": p2, "X": x, "true_tau": tau})


@dataclass
class BinnedConditionalCopula:
    """Piecewise-constant conditional copula indexed by conditioning quantiles."""

    family: str
    boundaries: FloatArray
    models: list[pv.Bicop]

    @property
    def n_bins(self) -> int:
        return len(self.models)

    def bin_indices(self, conditioning: FloatArray) -> NDArray[np.int64]:
        return np.clip(
            np.searchsorted(self.boundaries[1:-1], conditioning, side="right"),
            0,
            self.n_bins - 1,
        ).astype(int)

    def loglik(self, first: FloatArray, second: FloatArray, conditioning: FloatArray) -> FloatArray:
        indices = self.bin_indices(conditioning)
        output = np.empty(len(indices), dtype=float)
        for index, model in enumerate(self.models):
            mask = indices == index
            if np.any(mask):
                output[mask] = log_pdf(
                    model,
                    np.column_stack([first[mask], second[mask]]),
                )
        return output

    def tau_at(self, conditioning: FloatArray) -> FloatArray:
        indices = self.bin_indices(conditioning)
        taus = np.asarray([float(model.tau) for model in self.models])
        return taus[indices]


def fit_binned_conditional_copula(
    train: pd.DataFrame,
    family: str,
    n_bins: int,
    min_bin_size: int,
) -> BinnedConditionalCopula:
    """Fit equal-frequency piecewise copulas with one fixed selected family."""
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    boundaries = np.quantile(train["P2"].to_numpy(), quantiles)
    boundaries[0] = 0.0
    boundaries[-1] = 1.0
    # Repeated boundaries can occur only with ties; continuous synthetic data
    # should not have them, but fail clearly if the model is not estimable.
    if np.any(np.diff(boundaries) <= 0.0):
        raise ValueError("Conditioning quantile boundaries are not distinct.")
    indices = np.clip(
        np.searchsorted(boundaries[1:-1], train["P2"].to_numpy(), side="right"),
        0,
        n_bins - 1,
    )
    models: list[pv.Bicop] = []
    for bin_index in range(n_bins):
        subset = train.loc[indices == bin_index, ["P1", "X"]].to_numpy()
        if len(subset) < min_bin_size:
            raise ValueError(
                f"Bin {bin_index} has {len(subset)} rows; minimum is {min_bin_size}."
            )
        models.append(fit_fixed_family_bicop(subset, family))
    return BinnedConditionalCopula(
        family=normalize_family_name(family),
        boundaries=np.asarray(boundaries, dtype=float),
        models=models,
    )


def run_non_simplified_validation(
    *,
    sample_sizes: Sequence[int],
    seeds: Sequence[int],
    candidate_bins: Sequence[int],
    min_bin_size: int,
    controls: pv.FitControlsBicop,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare simplified and cross-validated binned conditional copulas."""
    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    for n in sample_sizes:
        for seed in seeds:
            data = simulate_non_simplified_conditional(n, seed)
            train_idx, valid_idx, test_idx = deterministic_split(n, seed)
            train = data.iloc[train_idx].reset_index(drop=True)
            validation = data.iloc[valid_idx].reset_index(drop=True)
            test = data.iloc[test_idx].reset_index(drop=True)

            simplified = fit_bicop(train[["P1", "X"]].to_numpy(), controls)
            selected_family = model_family_name(simplified)
            simplified_valid_ll = float(
                log_pdf(simplified, validation[["P1", "X"]].to_numpy()).mean()
            )
            candidate_records: list[tuple[BinnedConditionalCopula, float]] = []
            for n_bins in candidate_bins:
                try:
                    binned = fit_binned_conditional_copula(
                        train,
                        family=selected_family,
                        n_bins=int(n_bins),
                        min_bin_size=min_bin_size,
                    )
                except ValueError:
                    continue
                score = float(
                    binned.loglik(
                        validation["P1"].to_numpy(),
                        validation["X"].to_numpy(),
                        validation["P2"].to_numpy(),
                    ).mean()
                )
                candidate_records.append((binned, score))
            if not candidate_records:
                raise RuntimeError("No admissible binned conditional model was fitted.")
            selected_binned, selected_valid_ll = max(candidate_records, key=lambda item: item[1])

            simplified_test_ll = float(
                log_pdf(simplified, test[["P1", "X"]].to_numpy()).mean()
            )
            binned_test_ll = float(
                selected_binned.loglik(
                    test["P1"].to_numpy(),
                    test["X"].to_numpy(),
                    test["P2"].to_numpy(),
                ).mean()
            )
            true_tau = varying_tau(test["P2"].to_numpy())
            simplified_tau = np.full(len(test), float(simplified.tau))
            binned_tau = selected_binned.tau_at(test["P2"].to_numpy())
            rho = np.sin(0.5 * pi * true_tau)
            oracle_test_ll = float(
                gaussian_copula_log_density(
                    test["P1"].to_numpy(),
                    test["X"].to_numpy(),
                    rho,
                ).mean()
            )
            rows.append(
                {
                    "sample_size": int(n),
                    "seed": int(seed),
                    "simplified_family": selected_family,
                    "simplified_tau": float(simplified.tau),
                    "selected_bins": int(selected_binned.n_bins),
                    "simplified_validation_loglik_mean": simplified_valid_ll,
                    "binned_validation_loglik_mean": selected_valid_ll,
                    "simplified_test_loglik_mean": simplified_test_ll,
                    "binned_test_loglik_mean": binned_test_ll,
                    "oracle_test_loglik_mean": oracle_test_ll,
                    "binned_test_loglik_improvement": binned_test_ll - simplified_test_ll,
                    "simplified_tau_rmse": float(
                        np.sqrt(np.mean((simplified_tau - true_tau) ** 2))
                    ),
                    "binned_tau_rmse": float(
                        np.sqrt(np.mean((binned_tau - true_tau) ** 2))
                    ),
                }
            )
            grid = np.linspace(0.01, 0.99, 99)
            for conditioning, truth, binned_value in zip(
                grid,
                varying_tau(grid),
                selected_binned.tau_at(grid),
            ):
                curves.append(
                    {
                        "sample_size": int(n),
                        "seed": int(seed),
                        "conditioning_u2": float(conditioning),
                        "true_tau": float(truth),
                        "simplified_tau": float(simplified.tau),
                        "binned_tau": float(binned_value),
                        "selected_bins": int(selected_binned.n_bins),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(curves)
