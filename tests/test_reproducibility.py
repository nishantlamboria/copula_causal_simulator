from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


# Skip this test module cleanly if the principal project dependency is absent.
pytest.importorskip("pyvinecopulib")


from copula_causal_sim.copulas.conditional_vine_library import (
    build_conditional_vine_library,
)
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
from copula_causal_sim.generators.indegree_two import (
    IndegreeTwoGenerator,
    load_conditional_vine_metadata,
)
from copula_causal_sim.graphs.random_graphs import sample_random_dag


CLIP_EPS = 1e-6


def call_make_pseudo_observations(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Call make_pseudo_observations while supporting either `clip_eps`
    or `eps` as the clipping argument.
    """
    signature = inspect.signature(make_pseudo_observations)
    parameters = signature.parameters

    kwargs: dict[str, float] = {}

    if "clip_eps" in parameters:
        kwargs["clip_eps"] = CLIP_EPS
    elif "eps" in parameters:
        kwargs["eps"] = CLIP_EPS

    pseudo = make_pseudo_observations(dataframe, **kwargs)

    if not isinstance(pseudo, pd.DataFrame):
        raise TypeError(
            "make_pseudo_observations must return a pandas DataFrame."
        )

    return pseudo


def fit_marginal_library(
    dataframe: pd.DataFrame,
) -> EmpiricalMarginalLibrary:
    """
    Fit the empirical marginal library using the available public constructor.
    """
    candidate_methods = [
        "fit",
        "from_dataframe",
        "from_data",
    ]

    for method_name in candidate_methods:
        method = getattr(EmpiricalMarginalLibrary, method_name, None)

        if method is None:
            continue

        signature = inspect.signature(method)
        parameters = signature.parameters

        kwargs: dict[str, float] = {}

        if "clip_eps" in parameters:
            kwargs["clip_eps"] = CLIP_EPS
        elif "eps" in parameters:
            kwargs["eps"] = CLIP_EPS

        library = method(dataframe, **kwargs)

        if not isinstance(library, EmpiricalMarginalLibrary):
            raise TypeError(
                f"EmpiricalMarginalLibrary.{method_name}() must return "
                "an EmpiricalMarginalLibrary instance."
            )

        return library

    raise AttributeError(
        "EmpiricalMarginalLibrary must expose fit(), from_dataframe(), "
        "or from_data()."
    )


def save_library(library: Any, path: Path) -> None:
    """
    Save a fitted library using its available persistence method.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(library, "save_pickle"):
        library.save_pickle(path)
        return

    if hasattr(library, "save"):
        library.save(path)
        return

    raise AttributeError(
        f"{type(library).__name__} must expose save_pickle() or save()."
    )


def create_indegree_one_dataframe(
    n_rows: int = 350,
    seed: int = 123,
) -> pd.DataFrame:
    """
    Create a small continuous dataset with a clear parent-child dependence.
    """
    rng = np.random.default_rng(seed)

    parent = rng.normal(loc=0.0, scale=1.0, size=n_rows)
    noise = rng.normal(loc=0.0, scale=0.55, size=n_rows)
    child = 0.85 * parent + noise

    return pd.DataFrame(
        {
            "parent": parent,
            "child": child,
        }
    )


def create_indegree_two_dataframe(
    n_rows: int = 450,
    seed: int = 456,
) -> pd.DataFrame:
    """
    Create a small continuous three-variable dataset.

    The child depends on both parents, while the parents are also moderately
    dependent. This produces a meaningful two-parent conditional mechanism.
    """
    rng = np.random.default_rng(seed)

    parent_1 = rng.normal(loc=0.0, scale=1.0, size=n_rows)

    parent_2_noise = rng.normal(
        loc=0.0,
        scale=0.75,
        size=n_rows,
    )
    parent_2 = 0.45 * parent_1 + parent_2_noise

    child_noise = rng.normal(
        loc=0.0,
        scale=0.50,
        size=n_rows,
    )
    child = (
        0.90 * parent_1
        - 0.55 * parent_2
        + 0.25 * parent_1 * parent_2
        + child_noise
    )

    return pd.DataFrame(
        {
            "parent_1": parent_1,
            "parent_2": parent_2,
            "child": child,
        }
    )


@pytest.fixture(scope="module")
def indegree_one_generator(
    tmp_path_factory: pytest.TempPathFactory,
) -> IndegreeOneGenerator:
    """
    Build a temporary but genuine pair-copula generator.
    """
    root = tmp_path_factory.mktemp("reproducibility_indegree_one")

    dataframe = create_indegree_one_dataframe()
    pseudo = call_make_pseudo_observations(dataframe)
    marginal_library = fit_marginal_library(dataframe)

    pair_model_dir = root / "paircopula_models"

    pair_library = build_paircopula_library(
        u_df=pseudo,
        dataset_id="test_indegree_one",
        selection_criterion="bic",
        allow_rotations=True,
        num_threads=1,
        model_dir=pair_model_dir,
    )

    pair_metadata_path = root / "coupling_library.pkl"
    save_library(pair_library, pair_metadata_path)

    pair_metadata = load_paircopula_metadata(pair_metadata_path)

    return IndegreeOneGenerator(
        marginal_library=marginal_library,
        paircopula_metadata=pair_metadata,
        artifact_dir=root,
        clip_eps=CLIP_EPS,
    )


@pytest.fixture(scope="module")
def indegree_two_generator(
    tmp_path_factory: pytest.TempPathFactory,
) -> IndegreeTwoGenerator:
    """
    Build temporary genuine pair-copula and conditional D-vine libraries.
    """
    root = tmp_path_factory.mktemp("reproducibility_indegree_two")

    dataframe = create_indegree_two_dataframe()
    pseudo = call_make_pseudo_observations(dataframe)
    marginal_library = fit_marginal_library(dataframe)

    # Pair-copula library used for nodes with one parent.
    pair_model_dir = root / "paircopula_models"

    pair_library = build_paircopula_library(
        u_df=pseudo,
        dataset_id="test_indegree_two",
        selection_criterion="bic",
        allow_rotations=True,
        num_threads=1,
        model_dir=pair_model_dir,
    )

    pair_metadata_path = root / "coupling_library.pkl"
    save_library(pair_library, pair_metadata_path)

    pair_metadata = load_paircopula_metadata(pair_metadata_path)

    # Conditional D-vine library used for nodes with two parents.
    conditional_model_dir = root / "conditional_vines_indegree2"

    conditional_library = build_conditional_vine_library(
        u_df=pseudo,
        dataset_id="test_indegree_two",
        parent_set_size=2,
        selection_criterion="bic",
        allow_rotations=True,
        num_threads=1,
        model_dir=conditional_model_dir,
    )

    conditional_metadata_path = (
        root / "conditional_vine_library_indegree2.pkl"
    )
    save_library(
        conditional_library,
        conditional_metadata_path,
    )

    conditional_metadata = load_conditional_vine_metadata(
        conditional_metadata_path
    )

    relevant_records = [
        record
        for record in conditional_metadata["records"]
        if record.get("status") == "ok"
        and record.get("child") == "child"
        and {
            record.get("parent_1"),
            record.get("parent_2"),
        }
        == {"parent_1", "parent_2"}
    ]

    assert relevant_records, (
        "The temporary conditional library did not contain a successful "
        "child | parent_1, parent_2 mechanism."
    )

    return IndegreeTwoGenerator(
        marginal_library=marginal_library,
        paircopula_metadata=pair_metadata,
        conditional_vine_metadata=conditional_metadata,
        artifact_dir=root,
        clip_eps=CLIP_EPS,
    )


def test_random_dag_is_reproducible_for_same_seed() -> None:
    first = sample_random_dag(
        num_nodes=15,
        edge_prob=0.30,
        max_indegree=2,
        seed=987,
    )

    second = sample_random_dag(
        num_nodes=15,
        edge_prob=0.30,
        max_indegree=2,
        seed=987,
    )

    assert np.array_equal(first, second)


def test_indegree_one_same_seed_produces_identical_uniform_samples(
    indegree_one_generator: IndegreeOneGenerator,
) -> None:
    adjacency = np.array(
        [
            [0, 1],
            [0, 0],
        ],
        dtype=int,
    )

    first = indegree_one_generator.generate(
        adjacency=adjacency,
        n_samples=250,
        seed=101,
    )

    second = indegree_one_generator.generate(
        adjacency=adjacency,
        n_samples=250,
        seed=101,
    )

    assert_frame_equal(
        first.u,
        second.u,
        check_exact=True,
    )


def test_indegree_one_same_seed_produces_identical_original_scale_samples(
    indegree_one_generator: IndegreeOneGenerator,
) -> None:
    adjacency = np.array(
        [
            [0, 1],
            [0, 0],
        ],
        dtype=int,
    )

    first = indegree_one_generator.generate(
        adjacency=adjacency,
        n_samples=250,
        seed=102,
    )

    second = indegree_one_generator.generate(
        adjacency=adjacency,
        n_samples=250,
        seed=102,
    )

    assert_frame_equal(
        first.x,
        second.x,
        check_exact=True,
    )


def test_indegree_one_different_seeds_produce_different_samples(
    indegree_one_generator: IndegreeOneGenerator,
) -> None:
    adjacency = np.array(
        [
            [0, 1],
            [0, 0],
        ],
        dtype=int,
    )

    first = indegree_one_generator.generate(
        adjacency=adjacency,
        n_samples=250,
        seed=201,
    )

    second = indegree_one_generator.generate(
        adjacency=adjacency,
        n_samples=250,
        seed=202,
    )

    assert not np.array_equal(
        first.u.to_numpy(),
        second.u.to_numpy(),
    )


def test_indegree_one_seed_is_not_affected_by_previous_calls(
    indegree_one_generator: IndegreeOneGenerator,
) -> None:
    """
    The generator must create its randomness from the supplied seed rather
    than relying on mutable global NumPy state.
    """
    adjacency = np.array(
        [
            [0, 1],
            [0, 0],
        ],
        dtype=int,
    )

    expected = indegree_one_generator.generate(
        adjacency=adjacency,
        n_samples=200,
        seed=301,
    )

    # Intervening generation with a different seed must not alter the result
    # obtained when seed 301 is used again.
    indegree_one_generator.generate(
        adjacency=adjacency,
        n_samples=200,
        seed=999,
    )

    repeated = indegree_one_generator.generate(
        adjacency=adjacency,
        n_samples=200,
        seed=301,
    )

    assert_frame_equal(
        expected.u,
        repeated.u,
        check_exact=True,
    )
    assert_frame_equal(
        expected.x,
        repeated.x,
        check_exact=True,
    )


def test_indegree_two_same_seed_produces_identical_uniform_samples(
    indegree_two_generator: IndegreeTwoGenerator,
) -> None:
    # parent_1 -> child
    # parent_2 -> child
    adjacency = np.array(
        [
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 0],
        ],
        dtype=int,
    )

    first = indegree_two_generator.generate(
        adjacency=adjacency,
        n_samples=300,
        seed=401,
    )

    second = indegree_two_generator.generate(
        adjacency=adjacency,
        n_samples=300,
        seed=401,
    )

    assert_frame_equal(
        first.u,
        second.u,
        check_exact=True,
    )


def test_indegree_two_same_seed_produces_identical_original_scale_samples(
    indegree_two_generator: IndegreeTwoGenerator,
) -> None:
    adjacency = np.array(
        [
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 0],
        ],
        dtype=int,
    )

    first = indegree_two_generator.generate(
        adjacency=adjacency,
        n_samples=300,
        seed=402,
    )

    second = indegree_two_generator.generate(
        adjacency=adjacency,
        n_samples=300,
        seed=402,
    )

    assert_frame_equal(
        first.x,
        second.x,
        check_exact=True,
    )


def test_indegree_two_different_seeds_produce_different_samples(
    indegree_two_generator: IndegreeTwoGenerator,
) -> None:
    adjacency = np.array(
        [
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 0],
        ],
        dtype=int,
    )

    first = indegree_two_generator.generate(
        adjacency=adjacency,
        n_samples=300,
        seed=501,
    )

    second = indegree_two_generator.generate(
        adjacency=adjacency,
        n_samples=300,
        seed=502,
    )

    assert not np.array_equal(
        first.u.to_numpy(),
        second.u.to_numpy(),
    )


def test_indegree_two_seed_is_not_affected_by_previous_calls(
    indegree_two_generator: IndegreeTwoGenerator,
) -> None:
    adjacency = np.array(
        [
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 0],
        ],
        dtype=int,
    )

    expected = indegree_two_generator.generate(
        adjacency=adjacency,
        n_samples=225,
        seed=601,
    )

    indegree_two_generator.generate(
        adjacency=adjacency,
        n_samples=225,
        seed=999,
    )

    repeated = indegree_two_generator.generate(
        adjacency=adjacency,
        n_samples=225,
        seed=601,
    )

    assert_frame_equal(
        expected.u,
        repeated.u,
        check_exact=True,
    )
    assert_frame_equal(
        expected.x,
        repeated.x,
        check_exact=True,
    )


def test_indegree_two_generated_values_are_finite_and_inside_unit_interval(
    indegree_two_generator: IndegreeTwoGenerator,
) -> None:
    adjacency = np.array(
        [
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 0],
        ],
        dtype=int,
    )

    result = indegree_two_generator.generate(
        adjacency=adjacency,
        n_samples=300,
        seed=701,
    )

    u_values = result.u.to_numpy(dtype=float)
    x_values = result.x.to_numpy(dtype=float)

    assert np.isfinite(u_values).all()
    assert np.isfinite(x_values).all()

    assert (u_values > 0.0).all()
    assert (u_values < 1.0).all()