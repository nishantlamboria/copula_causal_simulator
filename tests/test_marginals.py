from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from copula_causal_sim.copulas.marginals import (
    EmpiricalMarginalLibrary,
    make_pseudo_observations,
)


CLIP_EPS = 1e-6


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """
    Small continuous dataset containing:

    - increasing and decreasing variables,
    - tied observations,
    - a constant variable,
    - non-default row indices.
    """
    return pd.DataFrame(
        {
            "increasing": [1.0, 2.0, 2.0, 4.0, 5.0, 7.0, 8.0, 10.0],
            "decreasing": [10.0, 8.0, 8.0, 6.0, 5.0, 3.0, 2.0, 1.0],
            "mixed": [0.5, -1.0, 3.0, 2.5, 2.5, 8.0, 4.0, 6.0],
            "constant": [4.2] * 8,
        },
        index=[10, 20, 30, 40, 50, 60, 70, 80],
    )


def call_make_pseudo_observations(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Call make_pseudo_observations while supporting either `clip_eps`
    or `eps` as the clipping-parameter name.
    """
    signature = inspect.signature(make_pseudo_observations)
    parameters = signature.parameters

    kwargs: dict[str, float] = {}

    if "clip_eps" in parameters:
        kwargs["clip_eps"] = CLIP_EPS
    elif "eps" in parameters:
        kwargs["eps"] = CLIP_EPS

    result = make_pseudo_observations(dataframe, **kwargs)

    if not isinstance(result, pd.DataFrame):
        raise TypeError(
            "make_pseudo_observations must return a pandas DataFrame."
        )

    return result


def fit_marginal_library(
    dataframe: pd.DataFrame,
) -> EmpiricalMarginalLibrary:
    """
    Construct the marginal library using its public fitting method.

    The helper supports the likely method names used during development,
    while failing clearly if none is available.
    """
    method_names = [
        "fit",
        "from_dataframe",
        "from_data",
    ]

    for method_name in method_names:
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
                f"EmpiricalMarginalLibrary.{method_name}() did not "
                "return an EmpiricalMarginalLibrary instance."
            )

        return library

    raise AttributeError(
        "EmpiricalMarginalLibrary must expose one of these fitting methods: "
        "fit(), from_dataframe(), or from_data()."
    )


def save_marginal_library(
    library: EmpiricalMarginalLibrary,
    path: Path,
) -> None:
    """
    Save the marginal library using the implemented public persistence method.
    """
    save_method = getattr(library, "save", None)
    if callable(save_method):
        save_method(path)
        return

    save_pickle_method = getattr(library, "save_pickle", None)
    if callable(save_pickle_method):
        save_pickle_method(path)
        return

    raise AttributeError(
        "EmpiricalMarginalLibrary must expose save() or save_pickle()."
    )


def load_marginal_library(
    path: Path,
) -> EmpiricalMarginalLibrary:
    """
    Load the marginal library using the implemented public persistence method.
    """
    load_method = getattr(EmpiricalMarginalLibrary, "load", None)
    if callable(load_method):
        loaded = load_method(path)
    else:
        load_pickle_method = getattr(EmpiricalMarginalLibrary, "load_pickle", None)
        if callable(load_pickle_method):
            loaded = load_pickle_method(path)
        else:
            raise AttributeError(
                "EmpiricalMarginalLibrary must expose load() or load_pickle()."
            )

    if not isinstance(loaded, EmpiricalMarginalLibrary):
        raise TypeError(
            "Loading the marginal library did not return an "
            "EmpiricalMarginalLibrary instance."
        )

    return loaded


def test_pseudo_observations_preserve_shape_and_columns(
    sample_dataframe: pd.DataFrame,
) -> None:
    pseudo = call_make_pseudo_observations(sample_dataframe)

    assert pseudo.shape == sample_dataframe.shape
    assert list(pseudo.columns) == list(sample_dataframe.columns)
    assert list(pseudo.index) == list(sample_dataframe.index)


def test_pseudo_observations_are_finite_and_strictly_inside_unit_interval(
    sample_dataframe: pd.DataFrame,
) -> None:
    pseudo = call_make_pseudo_observations(sample_dataframe)

    values = pseudo.to_numpy(dtype=float)

    assert np.isfinite(values).all()
    assert (values > 0.0).all()
    assert (values < 1.0).all()


def test_pseudo_observations_respect_clip_epsilon(
    sample_dataframe: pd.DataFrame,
) -> None:
    pseudo = call_make_pseudo_observations(sample_dataframe)

    values = pseudo.to_numpy(dtype=float)

    assert values.min() >= CLIP_EPS
    assert values.max() <= 1.0 - CLIP_EPS


def test_rank_order_is_preserved(
    sample_dataframe: pd.DataFrame,
) -> None:
    pseudo = call_make_pseudo_observations(sample_dataframe)

    increasing_order = sample_dataframe["increasing"].argsort().to_numpy()
    decreasing_order = sample_dataframe["decreasing"].argsort().to_numpy()

    increasing_u = pseudo["increasing"].to_numpy()[increasing_order]
    decreasing_u = pseudo["decreasing"].to_numpy()[decreasing_order]

    assert np.all(np.diff(increasing_u) >= 0.0)
    assert np.all(np.diff(decreasing_u) >= 0.0)


def test_tied_values_receive_equal_pseudo_observations(
    sample_dataframe: pd.DataFrame,
) -> None:
    pseudo = call_make_pseudo_observations(sample_dataframe)

    # Rows with index 20 and 30 both contain increasing=2.0.
    assert pseudo.loc[20, "increasing"] == pytest.approx(
        pseudo.loc[30, "increasing"]
    )

    # Rows with index 20 and 30 both contain decreasing=8.0.
    assert pseudo.loc[20, "decreasing"] == pytest.approx(
        pseudo.loc[30, "decreasing"]
    )

    # All observations in a constant column should obtain the same rank.
    assert pseudo["constant"].nunique() == 1


def test_pseudo_observation_transformation_is_deterministic(
    sample_dataframe: pd.DataFrame,
) -> None:
    first = call_make_pseudo_observations(sample_dataframe)
    second = call_make_pseudo_observations(sample_dataframe)

    assert_frame_equal(first, second)


def test_marginal_library_preserves_column_order(
    sample_dataframe: pd.DataFrame,
) -> None:
    library = fit_marginal_library(sample_dataframe)

    assert list(library.columns) == list(sample_dataframe.columns)


def test_inverse_transform_preserves_shape_and_columns(
    sample_dataframe: pd.DataFrame,
) -> None:
    library = fit_marginal_library(sample_dataframe)

    u_dataframe = pd.DataFrame(
        {
            column: np.linspace(0.1, 0.9, 7)
            for column in sample_dataframe.columns
        }
    )

    transformed = library.inverse_transform(u_dataframe)

    assert isinstance(transformed, pd.DataFrame)
    assert transformed.shape == u_dataframe.shape
    assert list(transformed.columns) == list(sample_dataframe.columns)


def test_inverse_transform_returns_only_finite_values(
    sample_dataframe: pd.DataFrame,
) -> None:
    library = fit_marginal_library(sample_dataframe)

    u_dataframe = pd.DataFrame(
        {
            column: [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
            for column in sample_dataframe.columns
        }
    )

    transformed = library.inverse_transform(u_dataframe)
    values = transformed.to_numpy(dtype=float)

    assert np.isfinite(values).all()
    assert not transformed.isna().any().any()


def test_inverse_transform_handles_boundary_values(
    sample_dataframe: pd.DataFrame,
) -> None:
    """
    The marginal inversion must safely handle values at or slightly outside
    the mathematical copula interval by clipping them internally.
    """
    library = fit_marginal_library(sample_dataframe)

    boundary_values = [
        -0.10,
        0.0,
        CLIP_EPS,
        0.5,
        1.0 - CLIP_EPS,
        1.0,
        1.10,
    ]

    u_dataframe = pd.DataFrame(
        {
            column: boundary_values
            for column in sample_dataframe.columns
        }
    )

    transformed = library.inverse_transform(u_dataframe)
    values = transformed.to_numpy(dtype=float)

    assert transformed.shape == u_dataframe.shape
    assert np.isfinite(values).all()
    assert not transformed.isna().any().any()


def test_inverse_transform_stays_inside_observed_data_range(
    sample_dataframe: pd.DataFrame,
) -> None:
    library = fit_marginal_library(sample_dataframe)

    u_dataframe = pd.DataFrame(
        {
            column: np.linspace(CLIP_EPS, 1.0 - CLIP_EPS, 101)
            for column in sample_dataframe.columns
        }
    )

    transformed = library.inverse_transform(u_dataframe)

    for column in sample_dataframe.columns:
        observed_minimum = sample_dataframe[column].min()
        observed_maximum = sample_dataframe[column].max()

        assert transformed[column].min() >= observed_minimum
        assert transformed[column].max() <= observed_maximum


def test_inverse_transform_is_monotone_in_probability(
    sample_dataframe: pd.DataFrame,
) -> None:
    """
    An empirical quantile function must be non-decreasing as U increases.
    """
    library = fit_marginal_library(sample_dataframe)

    probabilities = np.linspace(
        CLIP_EPS,
        1.0 - CLIP_EPS,
        101,
    )

    u_dataframe = pd.DataFrame(
        {
            column: probabilities
            for column in sample_dataframe.columns
        }
    )

    transformed = library.inverse_transform(u_dataframe)

    for column in sample_dataframe.columns:
        differences = np.diff(
            transformed[column].to_numpy(dtype=float)
        )

        assert np.all(differences >= -1e-12), (
            f"Inverse empirical marginal for '{column}' is not monotone."
        )


def test_constant_variable_is_inverse_transformed_correctly(
    sample_dataframe: pd.DataFrame,
) -> None:
    library = fit_marginal_library(sample_dataframe)

    u_dataframe = pd.DataFrame(
        {
            column: np.linspace(0.01, 0.99, 25)
            for column in sample_dataframe.columns
        }
    )

    transformed = library.inverse_transform(u_dataframe)

    assert np.allclose(
        transformed["constant"].to_numpy(dtype=float),
        4.2,
    )


def test_inverse_transform_is_deterministic(
    sample_dataframe: pd.DataFrame,
) -> None:
    library = fit_marginal_library(sample_dataframe)

    u_dataframe = pd.DataFrame(
        {
            column: np.linspace(0.05, 0.95, 20)
            for column in sample_dataframe.columns
        }
    )

    first = library.inverse_transform(u_dataframe)
    second = library.inverse_transform(u_dataframe)

    assert_frame_equal(first, second)


def test_marginal_library_save_load_roundtrip(
    sample_dataframe: pd.DataFrame,
    tmp_path: Path,
) -> None:
    library = fit_marginal_library(sample_dataframe)

    library_path = tmp_path / "marginals.pkl"
    save_marginal_library(library, library_path)

    assert library_path.exists()

    loaded_library = load_marginal_library(library_path)

    u_dataframe = pd.DataFrame(
        {
            column: np.linspace(0.05, 0.95, 20)
            for column in sample_dataframe.columns
        }
    )

    original_output = library.inverse_transform(u_dataframe)
    loaded_output = loaded_library.inverse_transform(u_dataframe)

    assert_frame_equal(original_output, loaded_output)