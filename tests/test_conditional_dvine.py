from __future__ import annotations

import inspect
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest


pv = pytest.importorskip("pyvinecopulib")


from copula_causal_sim.copulas.conditional_vine_library import (
    ConditionalVineLibrary,
    build_conditional_vine_library,
)
from copula_causal_sim.copulas.marginals import (
    make_pseudo_observations,
)
from copula_causal_sim.generators.indegree_two import (
    load_conditional_vine_metadata,
)


CLIP_EPS = 1e-6
ROUNDTRIP_TOLERANCE = 1e-6


def create_training_dataframe(
    n_rows: int = 700,
    seed: int = 123,
) -> pd.DataFrame:
    """
    Create a continuous three-variable dataset with a genuine
    two-parent conditional mechanism.

    The child depends on both parents and includes a mild nonlinear
    interaction.
    """
    rng = np.random.default_rng(seed)

    parent_1 = rng.normal(
        loc=0.0,
        scale=1.0,
        size=n_rows,
    )

    parent_2 = (
        0.40 * parent_1
        + rng.normal(
            loc=0.0,
            scale=0.80,
            size=n_rows,
        )
    )

    child = (
        0.85 * parent_1
        + 0.70 * parent_2
        + 0.20 * parent_1 * parent_2
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
    Call the current pseudo-observation implementation while supporting
    its possible clipping-argument name.
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


def save_library(
    library: ConditionalVineLibrary,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    library.save_pickle(path)


def select_stable_rows(
    pseudo_dataframe: pd.DataFrame,
    lower: float = 0.02,
    upper: float = 0.98,
) -> pd.DataFrame:
    """
    Exclude observations extremely close to zero or one when testing
    numerical inversion accuracy.

    Boundary behavior is handled separately by clipping in the generator.
    """
    mask = np.ones(
        len(pseudo_dataframe),
        dtype=bool,
    )

    for column in pseudo_dataframe.columns:
        values = pseudo_dataframe[column].to_numpy(
            dtype=float
        )

        mask &= values >= lower
        mask &= values <= upper

    selected = pseudo_dataframe.loc[mask].copy()

    assert len(selected) >= 100

    return selected


@pytest.fixture(scope="module")
def training_dataframe() -> pd.DataFrame:
    return create_training_dataframe()


@pytest.fixture(scope="module")
def pseudo_dataframe(
    training_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    return make_pseudo_dataframe(
        training_dataframe
    )


@pytest.fixture(scope="module")
def dvine_assets(
    tmp_path_factory: pytest.TempPathFactory,
    pseudo_dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Fit a real explicit three-variable D-vine and load its three stored
    bivariate copula components.
    """
    artifact_dir = tmp_path_factory.mktemp(
        "conditional_dvine"
    )

    model_dir = (
        artifact_dir
        / "conditional_vines_indegree2"
    )

    library = build_conditional_vine_library(
        u_df=pseudo_dataframe,
        dataset_id="test_conditional_dvine",
        parent_set_size=2,
        selection_criterion="bic",
        allow_rotations=True,
        num_threads=1,
        model_dir=model_dir,
    )

    metadata_path = (
        artifact_dir
        / "conditional_vine_library_indegree2.pkl"
    )

    save_library(
        library,
        metadata_path,
    )

    metadata = load_conditional_vine_metadata(
        metadata_path
    )

    matching_records = [
        record
        for record in metadata["records"]
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

    record = matching_records[0]

    c12_path = Path(
        record["bicop_parent_1_parent_2_file"]
    )

    c23_path = Path(
        record["bicop_parent_2_child_file"]
    )

    c13_given_2_path = Path(
        record[
            "bicop_parent_1_child_given_parent_2_file"
        ]
    )

    assert c12_path.exists()
    assert c23_path.exists()
    assert c13_given_2_path.exists()

    c12 = pv.Bicop.from_file(
        str(c12_path)
    )

    c23 = pv.Bicop.from_file(
        str(c23_path)
    )

    c13_given_2 = pv.Bicop.from_file(
        str(c13_given_2_path)
    )

    return {
        "artifact_dir": artifact_dir,
        "model_dir": model_dir,
        "library": library,
        "metadata": metadata,
        "metadata_path": metadata_path,
        "record": record,
        "c12": c12,
        "c23": c23,
        "c13_given_2": c13_given_2,
    }


def test_library_contains_expected_number_of_records(
    dvine_assets: dict[str, Any],
) -> None:
    """
    With three variables, each variable is used once as child and the
    remaining two form its parent set.
    """
    metadata = dvine_assets["metadata"]

    assert len(metadata["records"]) == 3


def test_selected_record_has_expected_structure(
    dvine_assets: dict[str, Any],
) -> None:
    record = dvine_assets["record"]

    assert record["dataset_id"] == "test_conditional_dvine"
    assert record["child"] == "child"

    assert {
        record["parent_1"],
        record["parent_2"],
    } == {
        "parent_1",
        "parent_2",
    }

    assert record["parent_set_size"] == 2
    assert record["n_variables"] == 3
    assert record["status"] == "ok"

    assert (
        record["model_format"]
        == "explicit_3d_dvine_bicop_json"
    )


def test_record_contains_all_three_bicop_files(
    dvine_assets: dict[str, Any],
) -> None:
    record = dvine_assets["record"]

    expected_keys = [
        "bicop_parent_1_parent_2_file",
        "bicop_parent_2_child_file",
        "bicop_parent_1_child_given_parent_2_file",
    ]

    for key in expected_keys:
        assert key in record
        assert Path(record[key]).exists()


def test_saved_metadata_identifies_explicit_hfunction_implementation(
    dvine_assets: dict[str, Any],
) -> None:
    metadata = dvine_assets["metadata"]

    assert (
        metadata["implementation"]
        == "explicit_3d_dvine_bicop_hfunctions"
    )


def test_parent_pair_hfunc1_hinv1_roundtrip(
    dvine_assets: dict[str, Any],
    pseudo_dataframe: pd.DataFrame,
) -> None:
    """
    C12 is fitted on:

        [parent_1, parent_2]

    Therefore:

        hfunc1(parent_1, parent_2)
            = F(parent_2 | parent_1)

        hinv1(parent_1, q)
            recovers parent_2.
    """
    stable = select_stable_rows(
        pseudo_dataframe
    )

    u1 = stable["parent_1"].to_numpy(
        dtype=float
    )

    u2 = stable["parent_2"].to_numpy(
        dtype=float
    )

    c12 = dvine_assets["c12"]

    data12 = np.asfortranarray(
        np.column_stack(
            [
                u1,
                u2,
            ]
        )
    )

    q2_given_1 = np.asarray(
        c12.hfunc1(data12),
        dtype=float,
    )

    inverse_input = np.asfortranarray(
        np.column_stack(
            [
                u1,
                q2_given_1,
            ]
        )
    )

    recovered_u2 = np.asarray(
        c12.hinv1(inverse_input),
        dtype=float,
    )

    assert np.isfinite(q2_given_1).all()
    assert np.isfinite(recovered_u2).all()

    assert np.allclose(
        recovered_u2,
        u2,
        atol=ROUNDTRIP_TOLERANCE,
        rtol=0.0,
    )


def test_parent2_child_hfunc1_hinv1_roundtrip(
    dvine_assets: dict[str, Any],
    pseudo_dataframe: pd.DataFrame,
) -> None:
    """
    C23 is fitted on:

        [parent_2, child]

    Therefore:

        hfunc1(parent_2, child)
            = F(child | parent_2)

        hinv1(parent_2, q)
            recovers child.
    """
    stable = select_stable_rows(
        pseudo_dataframe
    )

    u2 = stable["parent_2"].to_numpy(
        dtype=float
    )

    uy = stable["child"].to_numpy(
        dtype=float
    )

    c23 = dvine_assets["c23"]

    data23 = np.asfortranarray(
        np.column_stack(
            [
                u2,
                uy,
            ]
        )
    )

    wy = np.asarray(
        c23.hfunc1(data23),
        dtype=float,
    )

    inverse_input = np.asfortranarray(
        np.column_stack(
            [
                u2,
                wy,
            ]
        )
    )

    recovered_uy = np.asarray(
        c23.hinv1(inverse_input),
        dtype=float,
    )

    assert np.isfinite(wy).all()
    assert np.isfinite(recovered_uy).all()

    assert np.allclose(
        recovered_uy,
        uy,
        atol=ROUNDTRIP_TOLERANCE,
        rtol=0.0,
    )


def test_conditional_pair_hfunc1_hinv1_roundtrip(
    dvine_assets: dict[str, Any],
    pseudo_dataframe: pd.DataFrame,
) -> None:
    """
    C13|2 is fitted on:

        [
            F(parent_1 | parent_2),
            F(child | parent_2),
        ]

    Its hfunc1 and hinv1 operations must recover the second conditional
    coordinate.
    """
    stable = select_stable_rows(
        pseudo_dataframe
    )

    u1 = stable["parent_1"].to_numpy(
        dtype=float
    )

    u2 = stable["parent_2"].to_numpy(
        dtype=float
    )

    uy = stable["child"].to_numpy(
        dtype=float
    )

    c12 = dvine_assets["c12"]
    c23 = dvine_assets["c23"]
    c13_given_2 = dvine_assets[
        "c13_given_2"
    ]

    data12 = np.asfortranarray(
        np.column_stack(
            [
                u1,
                u2,
            ]
        )
    )

    data23 = np.asfortranarray(
        np.column_stack(
            [
                u2,
                uy,
            ]
        )
    )

    w1 = np.asarray(
        c12.hfunc2(data12),
        dtype=float,
    )

    wy = np.asarray(
        c23.hfunc1(data23),
        dtype=float,
    )

    w1 = np.clip(
        w1,
        CLIP_EPS,
        1.0 - CLIP_EPS,
    )

    wy = np.clip(
        wy,
        CLIP_EPS,
        1.0 - CLIP_EPS,
    )

    conditional_data = np.asfortranarray(
        np.column_stack(
            [
                w1,
                wy,
            ]
        )
    )

    q = np.asarray(
        c13_given_2.hfunc1(
            conditional_data
        ),
        dtype=float,
    )

    inverse_input = np.asfortranarray(
        np.column_stack(
            [
                w1,
                q,
            ]
        )
    )

    recovered_wy = np.asarray(
        c13_given_2.hinv1(
            inverse_input
        ),
        dtype=float,
    )

    assert np.isfinite(q).all()
    assert np.isfinite(recovered_wy).all()

    assert np.allclose(
        recovered_wy,
        wy,
        atol=ROUNDTRIP_TOLERANCE,
        rtol=0.0,
    )


def test_complete_dvine_conditional_roundtrip_recovers_child(
    dvine_assets: dict[str, Any],
    pseudo_dataframe: pd.DataFrame,
) -> None:
    """
    Test the complete mathematical path used by the indegree-two
    conditional sampler.

    Forward transform:

        w1 = F(parent_1 | parent_2)
        wy = F(child | parent_2)
        q  = F(wy | w1)

    Inverse transform:

        wy_recovered = F^{-1}(q | w1)
        child_recovered = F^{-1}(wy_recovered | parent_2)
    """
    stable = select_stable_rows(
        pseudo_dataframe
    )

    u1 = stable["parent_1"].to_numpy(
        dtype=float
    )

    u2 = stable["parent_2"].to_numpy(
        dtype=float
    )

    uy = stable["child"].to_numpy(
        dtype=float
    )

    c12 = dvine_assets["c12"]
    c23 = dvine_assets["c23"]
    c13_given_2 = dvine_assets[
        "c13_given_2"
    ]

    data12 = np.asfortranarray(
        np.column_stack(
            [
                u1,
                u2,
            ]
        )
    )

    data23 = np.asfortranarray(
        np.column_stack(
            [
                u2,
                uy,
            ]
        )
    )

    w1 = np.asarray(
        c12.hfunc2(data12),
        dtype=float,
    )

    wy = np.asarray(
        c23.hfunc1(data23),
        dtype=float,
    )

    w1 = np.clip(
        w1,
        CLIP_EPS,
        1.0 - CLIP_EPS,
    )

    wy = np.clip(
        wy,
        CLIP_EPS,
        1.0 - CLIP_EPS,
    )

    conditional_data = np.asfortranarray(
        np.column_stack(
            [
                w1,
                wy,
            ]
        )
    )

    q = np.asarray(
        c13_given_2.hfunc1(
            conditional_data
        ),
        dtype=float,
    )

    q = np.clip(
        q,
        CLIP_EPS,
        1.0 - CLIP_EPS,
    )

    recover_wy_input = np.asfortranarray(
        np.column_stack(
            [
                w1,
                q,
            ]
        )
    )

    recovered_wy = np.asarray(
        c13_given_2.hinv1(
            recover_wy_input
        ),
        dtype=float,
    )

    recovered_wy = np.clip(
        recovered_wy,
        CLIP_EPS,
        1.0 - CLIP_EPS,
    )

    recover_child_input = np.asfortranarray(
        np.column_stack(
            [
                u2,
                recovered_wy,
            ]
        )
    )

    recovered_child = np.asarray(
        c23.hinv1(
            recover_child_input
        ),
        dtype=float,
    )

    assert np.isfinite(recovered_child).all()

    assert np.allclose(
        recovered_child,
        uy,
        atol=ROUNDTRIP_TOLERANCE,
        rtol=0.0,
    )


def test_complete_roundtrip_is_deterministic(
    dvine_assets: dict[str, Any],
    pseudo_dataframe: pd.DataFrame,
) -> None:
    stable = select_stable_rows(
        pseudo_dataframe
    )

    u1 = stable["parent_1"].to_numpy(
        dtype=float
    )

    u2 = stable["parent_2"].to_numpy(
        dtype=float
    )

    uy = stable["child"].to_numpy(
        dtype=float
    )

    c12 = dvine_assets["c12"]
    c23 = dvine_assets["c23"]
    c13_given_2 = dvine_assets[
        "c13_given_2"
    ]

    data12 = np.asfortranarray(
        np.column_stack([u1, u2])
    )

    data23 = np.asfortranarray(
        np.column_stack([u2, uy])
    )

    first_w1 = np.asarray(
        c12.hfunc2(data12),
        dtype=float,
    )

    second_w1 = np.asarray(
        c12.hfunc2(data12),
        dtype=float,
    )

    first_wy = np.asarray(
        c23.hfunc1(data23),
        dtype=float,
    )

    second_wy = np.asarray(
        c23.hfunc1(data23),
        dtype=float,
    )

    assert np.array_equal(
        first_w1,
        second_w1,
    )

    assert np.array_equal(
        first_wy,
        second_wy,
    )

    conditional_data = np.asfortranarray(
        np.column_stack(
            [
                np.clip(
                    first_w1,
                    CLIP_EPS,
                    1.0 - CLIP_EPS,
                ),
                np.clip(
                    first_wy,
                    CLIP_EPS,
                    1.0 - CLIP_EPS,
                ),
            ]
        )
    )

    first_q = np.asarray(
        c13_given_2.hfunc1(
            conditional_data
        ),
        dtype=float,
    )

    second_q = np.asarray(
        c13_given_2.hfunc1(
            conditional_data
        ),
        dtype=float,
    )

    assert np.array_equal(
        first_q,
        second_q,
    )


def test_hfunction_outputs_are_valid_probabilities(
    dvine_assets: dict[str, Any],
    pseudo_dataframe: pd.DataFrame,
) -> None:
    stable = select_stable_rows(
        pseudo_dataframe
    )

    u1 = stable["parent_1"].to_numpy(
        dtype=float
    )

    u2 = stable["parent_2"].to_numpy(
        dtype=float
    )

    uy = stable["child"].to_numpy(
        dtype=float
    )

    c12 = dvine_assets["c12"]
    c23 = dvine_assets["c23"]

    data12 = np.asfortranarray(
        np.column_stack([u1, u2])
    )

    data23 = np.asfortranarray(
        np.column_stack([u2, uy])
    )

    w1 = np.asarray(
        c12.hfunc2(data12),
        dtype=float,
    )

    wy = np.asarray(
        c23.hfunc1(data23),
        dtype=float,
    )

    for values in (w1, wy):
        assert np.isfinite(values).all()
        assert (values >= 0.0).all()
        assert (values <= 1.0).all()


def test_empty_dataframe_is_rejected(
    tmp_path: Path,
) -> None:
    empty_dataframe = pd.DataFrame(
        columns=[
            "parent_1",
            "parent_2",
            "child",
        ]
    )

    with pytest.raises(ValueError):
        build_conditional_vine_library(
            u_df=empty_dataframe,
            dataset_id="empty_test",
            parent_set_size=2,
            model_dir=tmp_path,
        )


def test_unsupported_parent_set_size_is_rejected(
    pseudo_dataframe: pd.DataFrame,
    tmp_path: Path,
) -> None:
    with pytest.raises(NotImplementedError):
        build_conditional_vine_library(
            u_df=pseudo_dataframe,
            dataset_id="unsupported_test",
            parent_set_size=3,
            model_dir=tmp_path,
        )


def test_metadata_loader_rejects_wrong_parent_set_size(
    tmp_path: Path,
) -> None:
    invalid_payload = {
        "dataset_id": "invalid",
        "columns": [
            "parent_1",
            "parent_2",
            "child",
        ],
        "parent_set_size": 3,
        "records": [],
    }

    metadata_path = (
        tmp_path
        / "invalid_conditional_library.pkl"
    )

    with open(
        metadata_path,
        "wb",
    ) as file:
        pickle.dump(
            invalid_payload,
            file,
        )

    with pytest.raises(ValueError):
        load_conditional_vine_metadata(
            metadata_path
        )


def test_metadata_loader_rejects_missing_required_keys(
    tmp_path: Path,
) -> None:
    invalid_payload = {
        "dataset_id": "invalid",
        "columns": [
            "parent_1",
            "parent_2",
            "child",
        ],
    }

    metadata_path = (
        tmp_path
        / "missing_keys.pkl"
    )

    with open(
        metadata_path,
        "wb",
    ) as file:
        pickle.dump(
            invalid_payload,
            file,
        )

    with pytest.raises(ValueError):
        load_conditional_vine_metadata(
            metadata_path
        )


def test_metadata_loader_rejects_missing_file(
    tmp_path: Path,
) -> None:
    missing_path = (
        tmp_path
        / "does_not_exist.pkl"
    )

    with pytest.raises(FileNotFoundError):
        load_conditional_vine_metadata(
            missing_path
        )