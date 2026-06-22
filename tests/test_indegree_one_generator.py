from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kendalltau


pytest.importorskip("pyvinecopulib")


from copula_causal_sim.copulas.marginals import (
    EmpiricalMarginalLibrary,
    make_pseudo_observations,
)
from copula_causal_sim.copulas.paircopula_library import (
    build_paircopula_library,
)
from copula_causal_sim.generators.indegree_one import (
    IndegreeOneGenerator,
    load_paircopula_metadata,
)


CLIP_EPS = 1e-6


def create_training_dataframe(
    n_rows: int = 500,
    seed: int = 123,
) -> pd.DataFrame:
    """Create three continuous variables with clear positive dependence."""
    rng = np.random.default_rng(seed)

    root = rng.normal(size=n_rows)

    middle = (
        0.90 * root
        + rng.normal(scale=0.40, size=n_rows)
    )

    child = (
        0.75 * middle
        + rng.normal(scale=0.45, size=n_rows)
    )

    return pd.DataFrame(
        {
            "root": root,
            "middle": middle,
            "child": child,
        }
    )


def make_pseudo_dataframe(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Call the implemented pseudo-observation function."""
    signature = inspect.signature(make_pseudo_observations)
    kwargs: dict[str, float] = {}

    if "clip_eps" in signature.parameters:
        kwargs["clip_eps"] = CLIP_EPS
    elif "eps" in signature.parameters:
        kwargs["eps"] = CLIP_EPS

    result = make_pseudo_observations(dataframe, **kwargs)

    assert isinstance(result, pd.DataFrame)
    return result


def fit_marginal_library(
    dataframe: pd.DataFrame,
) -> EmpiricalMarginalLibrary:
    """Use the fitting constructor available in the current implementation."""
    for method_name in ("fit", "from_dataframe", "from_data"):
        method = getattr(
            EmpiricalMarginalLibrary,
            method_name,
            None,
        )

        if method is None:
            continue

        signature = inspect.signature(method)
        kwargs: dict[str, float] = {}

        if "clip_eps" in signature.parameters:
            kwargs["clip_eps"] = CLIP_EPS
        elif "eps" in signature.parameters:
            kwargs["eps"] = CLIP_EPS

        try:
            result = method(dataframe, **kwargs)
        except TypeError:
            continue

        if isinstance(result, EmpiricalMarginalLibrary):
            return result

    raise AttributeError(
        "Could not fit EmpiricalMarginalLibrary. Expected one of: "
        "fit(), from_dataframe(), or from_data()."
    )


def save_paircopula_library(
    library: Any,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(library, "save_pickle"):
        library.save_pickle(path)
    elif hasattr(library, "save"):
        library.save(path)
    else:
        raise AttributeError(
            "PairCopulaLibrary must provide save_pickle() or save()."
        )


@pytest.fixture(scope="module")
def training_dataframe() -> pd.DataFrame:
    return create_training_dataframe()


@pytest.fixture(scope="module")
def indegree_one_generator(
    tmp_path_factory: pytest.TempPathFactory,
    training_dataframe: pd.DataFrame,
) -> IndegreeOneGenerator:
    artifact_dir = tmp_path_factory.mktemp(
        "indegree_one_generator"
    )

    pseudo_dataframe = make_pseudo_dataframe(
        training_dataframe
    )

    marginal_library = fit_marginal_library(
        training_dataframe
    )

    pair_model_dir = artifact_dir / "paircopula_models"

    pair_library = build_paircopula_library(
        u_df=pseudo_dataframe,
        dataset_id="test_indegree_one",
        selection_criterion="bic",
        allow_rotations=True,
        num_threads=1,
        model_dir=pair_model_dir,
    )

    metadata_path = artifact_dir / "coupling_library.pkl"

    save_paircopula_library(
        pair_library,
        metadata_path,
    )

    metadata = load_paircopula_metadata(
        metadata_path
    )

    return IndegreeOneGenerator(
        marginal_library=marginal_library,
        paircopula_metadata=metadata,
        artifact_dir=artifact_dir,
        clip_eps=CLIP_EPS,
    )


def root_graph() -> np.ndarray:
    return np.zeros((3, 3), dtype=int)


def single_edge_graph() -> np.ndarray:
    # root -> middle
    return np.array(
        [
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=int,
    )


def chain_graph() -> np.ndarray:
    # root -> middle -> child
    return np.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ],
        dtype=int,
    )


def test_root_graph_has_correct_shape(
    indegree_one_generator: IndegreeOneGenerator,
) -> None:
    result = indegree_one_generator.generate(
        adjacency=root_graph(),
        n_samples=250,
        seed=10,
    )

    assert result.u.shape == (250, 3)
    assert result.x.shape == (250, 3)

    assert list(result.u.columns) == [
        "root",
        "middle",
        "child",
    ]

    assert list(result.x.columns) == [
        "root",
        "middle",
        "child",
    ]


def test_root_graph_produces_finite_uniform_values(
    indegree_one_generator: IndegreeOneGenerator,
) -> None:
    result = indegree_one_generator.generate(
        adjacency=root_graph(),
        n_samples=1000,
        seed=11,
    )

    values = result.u.to_numpy(dtype=float)

    assert np.isfinite(values).all()
    assert not np.isnan(values).any()

    assert (values > 0.0).all()
    assert (values < 1.0).all()

    assert values.min() >= CLIP_EPS
    assert values.max() <= 1.0 - CLIP_EPS


def test_root_graph_uniform_means_are_reasonable(
    indegree_one_generator: IndegreeOneGenerator,
) -> None:
    result = indegree_one_generator.generate(
        adjacency=root_graph(),
        n_samples=4000,
        seed=12,
    )

    means = result.u.mean().to_numpy(dtype=float)

    assert np.all(np.abs(means - 0.5) < 0.05)


def test_single_edge_generation_is_finite(
    indegree_one_generator: IndegreeOneGenerator,
) -> None:
    result = indegree_one_generator.generate(
        adjacency=single_edge_graph(),
        n_samples=500,
        seed=20,
    )

    assert np.isfinite(
        result.u.to_numpy(dtype=float)
    ).all()

    assert np.isfinite(
        result.x.to_numpy(dtype=float)
    ).all()

    assert not result.u.isna().any().any()
    assert not result.x.isna().any().any()


def test_single_edge_preserves_dependence_direction(
    indegree_one_generator: IndegreeOneGenerator,
) -> None:
    result = indegree_one_generator.generate(
        adjacency=single_edge_graph(),
        n_samples=2000,
        seed=21,
    )

    tau = kendalltau(
        result.u["root"],
        result.u["middle"],
    ).__getattribute__("statistic")

    assert tau is not None
    assert np.isfinite(tau)

    # The training data contains strong positive dependence.
    assert tau > 0.15


def test_disconnected_node_is_approximately_independent(
    indegree_one_generator: IndegreeOneGenerator,
) -> None:
    result = indegree_one_generator.generate(
        adjacency=single_edge_graph(),
        n_samples=3000,
        seed=22,
    )

    tau = kendalltau(
        result.u["middle"],
        result.u["child"],
    ).__getattribute__("statistic")

    assert tau is not None
    assert np.isfinite(tau)

    # The child is a separate root in this graph.
    assert abs(tau) < 0.10


def test_chain_graph_generates_all_nodes(
    indegree_one_generator: IndegreeOneGenerator,
) -> None:
    result = indegree_one_generator.generate(
        adjacency=chain_graph(),
        n_samples=750,
        seed=30,
    )

    assert result.u.shape == (750, 3)
    assert result.x.shape == (750, 3)

    assert np.isfinite(
        result.u.to_numpy(dtype=float)
    ).all()

    assert np.isfinite(
        result.x.to_numpy(dtype=float)
    ).all()


def test_chain_graph_preserves_local_edge_dependence(
    indegree_one_generator: IndegreeOneGenerator,
) -> None:
    result = indegree_one_generator.generate(
        adjacency=chain_graph(),
        n_samples=2000,
        seed=31,
    )

    tau_root_middle = kendalltau(
        result.u["root"],
        result.u["middle"],
    ).__getattribute__("statistic")

    tau_middle_child = kendalltau(
        result.u["middle"],
        result.u["child"],
    ).__getattribute__("statistic")

    assert tau_root_middle is not None
    assert tau_middle_child is not None

    assert np.isfinite(tau_root_middle)
    assert np.isfinite(tau_middle_child)

    assert tau_root_middle > 0.15
    assert tau_middle_child > 0.15


def test_original_scale_values_stay_in_observed_range(
    indegree_one_generator: IndegreeOneGenerator,
    training_dataframe: pd.DataFrame,
) -> None:
    result = indegree_one_generator.generate(
        adjacency=chain_graph(),
        n_samples=1000,
        seed=40,
    )

    tolerance = 1e-10

    for column in training_dataframe.columns:
        observed_min = float(
            training_dataframe[column].min()
        )
        observed_max = float(
            training_dataframe[column].max()
        )

        generated_min = float(
            result.x[column].min()
        )
        generated_max = float(
            result.x[column].max()
        )

        assert generated_min >= observed_min - tolerance
        assert generated_max <= observed_max + tolerance


def test_generation_result_contains_expected_metadata(
    indegree_one_generator: IndegreeOneGenerator,
) -> None:
    adjacency = chain_graph()

    result = indegree_one_generator.generate(
        adjacency=adjacency,
        n_samples=100,
        seed=55,
    )

    assert result.seed == 55

    assert result.columns == [
        "root",
        "middle",
        "child",
    ]

    assert np.array_equal(
        result.adjacency,
        adjacency,
    )

    # max_supported_indegree is optional in older result dataclasses.
    if hasattr(result, "max_supported_indegree"):
        assert result.__getattribute__("max_supported_indegree") == 1


def test_same_seed_is_reproducible(
    indegree_one_generator: IndegreeOneGenerator,
) -> None:
    first = indegree_one_generator.generate(
        adjacency=chain_graph(),
        n_samples=300,
        seed=70,
    )

    second = indegree_one_generator.generate(
        adjacency=chain_graph(),
        n_samples=300,
        seed=70,
    )

    assert np.array_equal(
        first.u.to_numpy(),
        second.u.to_numpy(),
    )

    assert np.array_equal(
        first.x.to_numpy(),
        second.x.to_numpy(),
    )


def test_different_seeds_change_generated_values(
    indegree_one_generator: IndegreeOneGenerator,
) -> None:
    first = indegree_one_generator.generate(
        adjacency=chain_graph(),
        n_samples=300,
        seed=71,
    )

    second = indegree_one_generator.generate(
        adjacency=chain_graph(),
        n_samples=300,
        seed=72,
    )

    assert not np.array_equal(
        first.u.to_numpy(),
        second.u.to_numpy(),
    )


def test_nonpositive_sample_count_is_rejected(
    indegree_one_generator: IndegreeOneGenerator,
) -> None:
    with pytest.raises(ValueError):
        indegree_one_generator.generate(
            adjacency=root_graph(),
            n_samples=0,
            seed=60,
        )


def test_wrong_graph_size_is_rejected(
    indegree_one_generator: IndegreeOneGenerator,
) -> None:
    wrong_size_graph = np.array(
        [
            [0, 1],
            [0, 0],
        ],
        dtype=int,
    )

    with pytest.raises(ValueError):
        indegree_one_generator.generate(
            adjacency=wrong_size_graph,
            n_samples=100,
            seed=61,
        )


def test_indegree_two_graph_is_rejected(
    indegree_one_generator: IndegreeOneGenerator,
) -> None:
    invalid_graph = np.array(
        [
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 0],
        ],
        dtype=int,
    )

    with pytest.raises(ValueError):
        indegree_one_generator.generate(
            adjacency=invalid_graph,
            n_samples=100,
            seed=62,
        )


def test_cyclic_graph_is_rejected(
    indegree_one_generator: IndegreeOneGenerator,
) -> None:
    cyclic_graph = np.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ],
        dtype=int,
    )

    with pytest.raises(ValueError):
        indegree_one_generator.generate(
            adjacency=cyclic_graph,
            n_samples=100,
            seed=63,
        )