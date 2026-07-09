from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kendalltau


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
    load_paircopula_metadata,
)
from copula_causal_sim.generators.indegree_two import (
    IndegreeTwoGenerator,
    load_conditional_vine_metadata,
)


CLIP_EPS = 1e-6


def create_training_dataframe(
    n_rows: int = 600,
    seed: int = 123,
) -> pd.DataFrame:
    """
    Create a three-variable continuous dataset with a genuine
    two-parent mechanism.

    The child depends positively on both parent variables.
    """
    rng = np.random.default_rng(seed)

    parent_1 = rng.normal(
        loc=0.0,
        scale=1.0,
        size=n_rows,
    )

    parent_2 = (
        0.35 * parent_1
        + rng.normal(
            loc=0.0,
            scale=0.85,
            size=n_rows,
        )
    )

    child = (
        0.90 * parent_1
        + 0.70 * parent_2
        + 0.15 * parent_1 * parent_2
        + rng.normal(
            loc=0.0,
            scale=0.45,
            size=n_rows,
        )
    )

    return pd.DataFrame(
        {
            "parent_1": parent_1,
            "parent_2": parent_2,
            "child": child,
        }
    )


def make_pseudo_dataframe(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Call the implemented pseudo-observation function using its
    actual clipping argument when present.
    """
    signature = inspect.signature(make_pseudo_observations)
    kwargs: dict[str, float] = {}

    if "clip_eps" in signature.parameters:
        kwargs["clip_eps"] = CLIP_EPS
    elif "eps" in signature.parameters:
        kwargs["eps"] = CLIP_EPS

    result = make_pseudo_observations(
        dataframe,
        **kwargs,
    )

    assert isinstance(result, pd.DataFrame)

    return result


def fit_marginal_library(
    dataframe: pd.DataFrame,
) -> EmpiricalMarginalLibrary:
    """
    Use the fitting constructor available in the current marginal
    implementation.
    """
    for method_name in (
        "fit",
        "from_dataframe",
        "from_data",
    ):
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
            result = method(
                dataframe,
                **kwargs,
            )
        except TypeError:
            continue

        if isinstance(
            result,
            EmpiricalMarginalLibrary,
        ):
            return result

    raise AttributeError(
        "Could not fit EmpiricalMarginalLibrary. Expected one of: "
        "fit(), from_dataframe(), or from_data()."
    )


def save_library(
    library: Any,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if hasattr(library, "save_pickle"):
        library.save_pickle(path)
        return

    if hasattr(library, "save"):
        library.save(path)
        return

    raise AttributeError(
        f"{type(library).__name__} must provide "
        "save_pickle() or save()."
    )


@pytest.fixture(scope="module")
def training_dataframe() -> pd.DataFrame:
    return create_training_dataframe()


@pytest.fixture(scope="module")
def indegree_two_generator(
    tmp_path_factory: pytest.TempPathFactory,
    training_dataframe: pd.DataFrame,
) -> IndegreeTwoGenerator:
    """
    Build real temporary pair-copula and conditional D-vine libraries.

    No Sachs, diabetes, breast-cancer, or previously generated artifact
    is required.
    """
    artifact_dir = tmp_path_factory.mktemp(
        "indegree_two_generator"
    )

    pseudo_dataframe = make_pseudo_dataframe(
        training_dataframe
    )

    marginal_library = fit_marginal_library(
        training_dataframe
    )

    # --------------------------------------------------------------
    # Pair-copula library for one-parent mechanisms
    # --------------------------------------------------------------
    pair_model_dir = (
        artifact_dir
        / "paircopula_models"
    )

    pair_library = build_paircopula_library(
        u_df=pseudo_dataframe,
        dataset_id="test_indegree_two",
        selection_criterion="bic",
        allow_rotations=True,
        num_threads=1,
        model_dir=pair_model_dir,
    )

    pair_metadata_path = (
        artifact_dir
        / "coupling_library.pkl"
    )

    save_library(
        pair_library,
        pair_metadata_path,
    )

    pair_metadata = load_paircopula_metadata(
        pair_metadata_path
    )

    # --------------------------------------------------------------
    # Explicit conditional D-vine library
    # --------------------------------------------------------------
    conditional_model_dir = (
        artifact_dir
        / "conditional_vines_indegree2"
    )

    conditional_library = build_conditional_vine_library(
        u_df=pseudo_dataframe,
        dataset_id="test_indegree_two",
        parent_set_size=2,
        selection_criterion="bic",
        allow_rotations=True,
        num_threads=1,
        model_dir=conditional_model_dir,
    )

    conditional_metadata_path = (
        artifact_dir
        / "conditional_vine_library_indegree2.pkl"
    )

    conditional_library.save_pickle(
        conditional_metadata_path
    )

    conditional_metadata = load_conditional_vine_metadata(
        conditional_metadata_path
    )

    matching_records = [
        record
        for record in conditional_metadata["records"]
        if record.get("status") == "ok"
        and record.get("child") == "child"
        and {
            record.get("parent_1"),
            record.get("parent_2"),
        }
        == {
            "parent_1",
            "parent_2",
        }
    ]

    assert len(matching_records) == 1

    return IndegreeTwoGenerator(
        marginal_library=marginal_library,
        paircopula_metadata=pair_metadata,
        conditional_vine_metadata=conditional_metadata,
        artifact_dir=artifact_dir,
        clip_eps=CLIP_EPS,
    )


def root_graph() -> np.ndarray:
    return np.zeros(
        shape=(3, 3),
        dtype=int,
    )


def one_parent_graph() -> np.ndarray:
    # parent_1 -> child
    return np.array(
        [
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=int,
    )


def two_parent_graph() -> np.ndarray:
    # parent_1 -> child
    # parent_2 -> child
    return np.array(
        [
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 0],
        ],
        dtype=int,
    )


def chain_graph() -> np.ndarray:
    # parent_1 -> parent_2 -> child
    return np.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ],
        dtype=int,
    )


def test_root_graph_has_correct_shape_and_columns(
    indegree_two_generator: IndegreeTwoGenerator,
) -> None:
    result = indegree_two_generator.generate(
        adjacency=root_graph(),
        n_samples=250,
        seed=10,
    )

    assert result.u.shape == (250, 3)
    assert result.x.shape == (250, 3)

    assert list(result.u.columns) == [
        "parent_1",
        "parent_2",
        "child",
    ]

    assert list(result.x.columns) == [
        "parent_1",
        "parent_2",
        "child",
    ]


def test_two_parent_graph_generates_finite_values(
    indegree_two_generator: IndegreeTwoGenerator,
) -> None:
    result = indegree_two_generator.generate(
        adjacency=two_parent_graph(),
        n_samples=750,
        seed=20,
    )

    u_values = result.u.to_numpy(dtype=float)
    x_values = result.x.to_numpy(dtype=float)

    assert np.isfinite(u_values).all()
    assert np.isfinite(x_values).all()

    assert not result.u.isna().any().any()
    assert not result.x.isna().any().any()


def test_generated_uniform_values_are_inside_open_unit_interval(
    indegree_two_generator: IndegreeTwoGenerator,
) -> None:
    result = indegree_two_generator.generate(
        adjacency=two_parent_graph(),
        n_samples=1000,
        seed=21,
    )

    values = result.u.to_numpy(dtype=float)

    assert (values > 0.0).all()
    assert (values < 1.0).all()

    assert values.min() >= CLIP_EPS
    assert values.max() <= 1.0 - CLIP_EPS


def test_one_parent_dispatch_works_inside_indegree_two_generator(
    indegree_two_generator: IndegreeTwoGenerator,
) -> None:
    """
    The indegree-two generator must continue to use the pair-copula
    implementation for nodes having exactly one parent.
    """
    result = indegree_two_generator.generate(
        adjacency=one_parent_graph(),
        n_samples=1500,
        seed=30,
    )

    tau = kendalltau(
        result.u["parent_1"],
        result.u["child"],
    ).__getattribute__("statistic")

    assert tau is not None
    assert np.isfinite(tau)
    assert tau > 0.10


def test_two_parent_mechanism_preserves_parent_1_dependence(
    indegree_two_generator: IndegreeTwoGenerator,
) -> None:
    result = indegree_two_generator.generate(
        adjacency=two_parent_graph(),
        n_samples=2000,
        seed=31,
    )

    tau = kendalltau(
        result.u["parent_1"],
        result.u["child"],
    ).__getattribute__("statistic")

    assert tau is not None
    assert np.isfinite(tau)
    assert tau > 0.10


def test_two_parent_mechanism_preserves_parent_2_dependence(
    indegree_two_generator: IndegreeTwoGenerator,
) -> None:
    result = indegree_two_generator.generate(
        adjacency=two_parent_graph(),
        n_samples=2000,
        seed=32,
    )

    tau = kendalltau(
        result.u["parent_2"],
        result.u["child"],
    ).__getattribute__("statistic")

    assert tau is not None
    assert np.isfinite(tau)
    assert tau > 0.10


def test_chain_graph_uses_one_parent_mechanisms(
    indegree_two_generator: IndegreeTwoGenerator,
) -> None:
    result = indegree_two_generator.generate(
        adjacency=chain_graph(),
        n_samples=1800,
        seed=33,
    )

    tau_parent_chain = kendalltau(
        result.u["parent_1"],
        result.u["parent_2"],
    ).__getattribute__("statistic")

    tau_child_chain = kendalltau(
        result.u["parent_2"],
        result.u["child"],
    ).__getattribute__("statistic")

    assert tau_parent_chain is not None
    assert tau_child_chain is not None

    assert np.isfinite(tau_parent_chain)
    assert np.isfinite(tau_child_chain)

    assert tau_parent_chain > 0.05
    assert tau_child_chain > 0.10


def test_original_scale_values_remain_in_training_ranges(
    indegree_two_generator: IndegreeTwoGenerator,
    training_dataframe: pd.DataFrame,
) -> None:
    result = indegree_two_generator.generate(
        adjacency=two_parent_graph(),
        n_samples=1000,
        seed=40,
    )

    tolerance = 1e-10

    for column in training_dataframe.columns:
        observed_minimum = float(
            training_dataframe[column].min()
        )

        observed_maximum = float(
            training_dataframe[column].max()
        )

        generated_minimum = float(
            result.x[column].min()
        )

        generated_maximum = float(
            result.x[column].max()
        )

        assert generated_minimum >= observed_minimum - tolerance
        assert generated_maximum <= observed_maximum + tolerance


def test_generation_result_has_exact_expected_metadata(
    indegree_two_generator: IndegreeTwoGenerator,
) -> None:
    adjacency = two_parent_graph()

    result = indegree_two_generator.generate(
        adjacency=adjacency,
        n_samples=100,
        seed=50,
    )

    assert result.__getattribute__("seed") == 50

    assert result.__getattribute__("columns") == [
        "parent_1",
        "parent_2",
        "child",
    ]

    assert result.__getattribute__(
        "max_supported_indegree"
    ) == 2

    assert np.array_equal(
        result.__getattribute__("adjacency"),
        adjacency,
    )


def test_same_seed_produces_identical_samples(
    indegree_two_generator: IndegreeTwoGenerator,
) -> None:
    first = indegree_two_generator.generate(
        adjacency=two_parent_graph(),
        n_samples=400,
        seed=60,
    )

    second = indegree_two_generator.generate(
        adjacency=two_parent_graph(),
        n_samples=400,
        seed=60,
    )

    assert np.array_equal(
        first.u.to_numpy(),
        second.u.to_numpy(),
    )

    assert np.array_equal(
        first.x.to_numpy(),
        second.x.to_numpy(),
    )


def test_different_seeds_produce_different_samples(
    indegree_two_generator: IndegreeTwoGenerator,
) -> None:
    first = indegree_two_generator.generate(
        adjacency=two_parent_graph(),
        n_samples=400,
        seed=61,
    )

    second = indegree_two_generator.generate(
        adjacency=two_parent_graph(),
        n_samples=400,
        seed=62,
    )

    assert not np.array_equal(
        first.u.to_numpy(),
        second.u.to_numpy(),
    )


def test_conditional_record_lookup_accepts_reversed_parent_order(
    indegree_two_generator: IndegreeTwoGenerator,
) -> None:
    first = indegree_two_generator._get_conditional_record(
        child_name="child",
        parent_names=[
            "parent_1",
            "parent_2",
        ],
    )

    second = indegree_two_generator._get_conditional_record(
        child_name="child",
        parent_names=[
            "parent_2",
            "parent_1",
        ],
    )

    assert first is second

    assert first["child"] == "child"

    assert {
        first["parent_1"],
        first["parent_2"],
    } == {
        "parent_1",
        "parent_2",
    }


def test_conditional_record_contains_expected_model_files(
    indegree_two_generator: IndegreeTwoGenerator,
) -> None:
    record = indegree_two_generator._get_conditional_record(
        child_name="child",
        parent_names=[
            "parent_1",
            "parent_2",
        ],
    )

    required_file_keys = [
        "bicop_parent_1_parent_2_file",
        "bicop_parent_2_child_file",
        "bicop_parent_1_child_given_parent_2_file",
    ]

    for key in required_file_keys:
        assert key in record
        assert Path(record[key]).exists()


def test_bicop_models_are_cached(
    indegree_two_generator: IndegreeTwoGenerator,
) -> None:
    record = indegree_two_generator._get_conditional_record(
        child_name="child",
        parent_names=[
            "parent_1",
            "parent_2",
        ],
    )

    model_path = record[
        "bicop_parent_1_parent_2_file"
    ]

    first = indegree_two_generator._load_bicop(
        model_path
    )

    second = indegree_two_generator._load_bicop(
        model_path
    )

    assert first is second


def test_wrong_number_of_parent_names_is_rejected(
    indegree_two_generator: IndegreeTwoGenerator,
) -> None:
    with pytest.raises(ValueError):
        indegree_two_generator._get_conditional_record(
            child_name="child",
            parent_names=["parent_1"],
        )


def test_missing_conditional_mechanism_is_rejected(
    indegree_two_generator: IndegreeTwoGenerator,
) -> None:
    with pytest.raises(KeyError):
        indegree_two_generator._get_conditional_record(
            child_name="child",
            parent_names=[
                "parent_1",
                "unknown_parent",
            ],
        )


def test_nonpositive_sample_count_is_rejected(
    indegree_two_generator: IndegreeTwoGenerator,
) -> None:
    with pytest.raises(ValueError):
        indegree_two_generator.generate(
            adjacency=root_graph(),
            n_samples=0,
            seed=70,
        )


def test_wrong_graph_size_is_rejected(
    indegree_two_generator: IndegreeTwoGenerator,
) -> None:
    wrong_size_graph = np.array(
        [
            [0, 1],
            [0, 0],
        ],
        dtype=int,
    )

    with pytest.raises(ValueError):
        indegree_two_generator.generate(
            adjacency=wrong_size_graph,
            n_samples=100,
            seed=71,
        )


def test_cyclic_graph_is_rejected(
    indegree_two_generator: IndegreeTwoGenerator,
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
        indegree_two_generator.generate(
            adjacency=cyclic_graph,
            n_samples=100,
            seed=72,
        )

def test_generator_record_order_matches_selected_metadata(
    indegree_two_generator: IndegreeTwoGenerator,
) -> None:
    record = indegree_two_generator._get_conditional_record(
        child_name="child",
        parent_names=["parent_1", "parent_2"],
    )

    selected_order = indegree_two_generator._selected_order_from_record(
        record
    )

    assert selected_order == [
        record["parent_1"],
        record["parent_2"],
        record["child"],
    ]
    assert set(selected_order[:2]) == {"parent_1", "parent_2"}


def test_old_fixed_order_record_remains_loadable() -> None:
    old_record = {
        "status": "ok",
        "child": "child",
        "parent_1": "parent_2",
        "parent_2": "parent_1",
    }

    lookup = IndegreeTwoGenerator._build_conditional_lookup([old_record])
    record = lookup[("child", frozenset({"parent_1", "parent_2"}))]

    assert IndegreeTwoGenerator._selected_order_from_record(record) == [
        "parent_2",
        "parent_1",
        "child",
    ]


def test_inconsistent_selected_order_is_rejected() -> None:
    record = {
        "status": "ok",
        "child": "child",
        "parent_1": "parent_1",
        "parent_2": "parent_2",
        "selected_order": ["parent_2", "parent_1", "child"],
    }

    with pytest.raises(ValueError, match="inconsistent selected-order"):
        IndegreeTwoGenerator._selected_order_from_record(record)


def test_duplicate_parent_set_records_are_rejected() -> None:
    records = [
        {
            "status": "ok",
            "child": "child",
            "parent_1": "parent_1",
            "parent_2": "parent_2",
        },
        {
            "status": "ok",
            "child": "child",
            "parent_1": "parent_2",
            "parent_2": "parent_1",
        },
    ]

    with pytest.raises(ValueError, match="Duplicate conditional D-vine"):
        IndegreeTwoGenerator._build_conditional_lookup(records)
