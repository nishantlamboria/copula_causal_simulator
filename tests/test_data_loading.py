from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from copula_causal_sim.config import load_config
from copula_causal_sim.data.loaders import (
    load_tabular_dataset,
    make_dataset_summary,
    validate_numeric_dataframe,
)


def make_dataset_config(
    path: Path,
    file_type: str = "csv",
    continuous_columns: str | list[str] = "all",
    drop_columns: list[str] | None = None,
    missing_policy: str = "error",
) -> dict:
    """
    Construct the exact configuration structure expected by
    load_tabular_dataset().
    """
    return {
        "dataset_id": "test_dataset",
        "path": str(path),
        "file_type": file_type,
        "data": {
            "continuous_columns": continuous_columns,
            "drop_columns": (
                [] if drop_columns is None else drop_columns
            ),
            "missing_policy": missing_policy,
            "clip_eps": 1e-6,
        },
        "copula": {
            "selection_criterion": "bic",
            "allow_rotations": True,
            "num_threads": 1,
            "families": [
                "indep",
                "gaussian",
                "student",
                "clayton",
                "gumbel",
                "frank",
            ],
        },
        "outputs": {
            "artifact_dir": "artifacts/test",
            "table_dir": "reports/test",
        },
    }


# ---------------------------------------------------------------------
# validate_numeric_dataframe
# ---------------------------------------------------------------------


def test_validate_numeric_dataframe_accepts_valid_numeric_data() -> None:
    dataframe = pd.DataFrame(
        {
            "integer_column": [1, 2, 3],
            "float_column": [1.5, 2.5, 3.5],
        }
    )

    # The function returns None when validation succeeds.
    result = validate_numeric_dataframe(dataframe)

    assert result is None


def test_validate_numeric_dataframe_rejects_empty_dataframe() -> None:
    dataframe = pd.DataFrame()

    with pytest.raises(
        ValueError,
        match="empty",
    ):
        validate_numeric_dataframe(dataframe)


def test_validate_numeric_dataframe_rejects_non_numeric_columns() -> None:
    dataframe = pd.DataFrame(
        {
            "numeric": [1.0, 2.0, 3.0],
            "category": ["a", "b", "c"],
        }
    )

    with pytest.raises(
        ValueError,
        match="Non-numeric",
    ):
        validate_numeric_dataframe(dataframe)


def test_validate_numeric_dataframe_reports_non_numeric_column_name() -> None:
    dataframe = pd.DataFrame(
        {
            "numeric": [1.0, 2.0, 3.0],
            "invalid_column": ["x", "y", "z"],
        }
    )

    with pytest.raises(ValueError) as error:
        validate_numeric_dataframe(dataframe)

    assert "invalid_column" in str(error.value)


def test_validate_numeric_dataframe_rejects_missing_values() -> None:
    dataframe = pd.DataFrame(
        {
            "first": [1.0, np.nan, 3.0],
            "second": [4.0, 5.0, 6.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="Missing values",
    ):
        validate_numeric_dataframe(dataframe)


def test_validate_numeric_dataframe_reports_missing_column_name() -> None:
    dataframe = pd.DataFrame(
        {
            "complete": [1.0, 2.0, 3.0],
            "contains_missing": [4.0, np.nan, 6.0],
        }
    )

    with pytest.raises(ValueError) as error:
        validate_numeric_dataframe(dataframe)

    assert "contains_missing" in str(error.value)


# ---------------------------------------------------------------------
# load_tabular_dataset: supported file formats
# ---------------------------------------------------------------------


def test_load_csv_dataset(
    tmp_path: Path,
) -> None:
    expected = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0],
        }
    )

    dataset_path = tmp_path / "dataset.csv"
    expected.to_csv(dataset_path, index=False)

    config = make_dataset_config(
        path=dataset_path,
        file_type="csv",
    )

    loaded = load_tabular_dataset(config)

    pd.testing.assert_frame_equal(
        loaded,
        expected,
        check_dtype=False,
    )


def test_load_tsv_dataset(
    tmp_path: Path,
) -> None:
    expected = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0],
        }
    )

    dataset_path = tmp_path / "dataset.tsv"
    expected.to_csv(
        dataset_path,
        sep="\t",
        index=False,
    )

    config = make_dataset_config(
        path=dataset_path,
        file_type="tsv",
    )

    loaded = load_tabular_dataset(config)

    pd.testing.assert_frame_equal(
        loaded,
        expected,
        check_dtype=False,
    )


def test_load_whitespace_separated_dataset(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.txt"

    dataset_path.write_text(
        "a b c\n"
        "1.0 2.0 3.0\n"
        "4.0 5.0 6.0\n"
        "7.0 8.0 9.0\n",
        encoding="utf-8",
    )

    config = make_dataset_config(
        path=dataset_path,
        file_type="whitespace_csv",
    )

    loaded = load_tabular_dataset(config)

    expected = pd.DataFrame(
        {
            "a": [1.0, 4.0, 7.0],
            "b": [2.0, 5.0, 8.0],
            "c": [3.0, 6.0, 9.0],
        }
    )

    pd.testing.assert_frame_equal(
        loaded,
        expected,
        check_dtype=False,
    )


def test_load_dataset_rejects_missing_file(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "does_not_exist.csv"

    config = make_dataset_config(
        path=missing_path,
        file_type="csv",
    )

    with pytest.raises(
        FileNotFoundError,
        match="Dataset file not found",
    ):
        load_tabular_dataset(config)


def test_load_dataset_rejects_unsupported_file_type(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.csv"

    pd.DataFrame(
        {
            "a": [1.0, 2.0],
        }
    ).to_csv(dataset_path, index=False)

    config = make_dataset_config(
        path=dataset_path,
        file_type="excel",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported file_type",
    ):
        load_tabular_dataset(config)


# ---------------------------------------------------------------------
# load_tabular_dataset: column handling
# ---------------------------------------------------------------------


def test_load_dataset_drops_requested_columns(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "keep_1": [1.0, 2.0, 3.0],
            "remove_me": [10.0, 20.0, 30.0],
            "keep_2": [4.0, 5.0, 6.0],
        }
    )

    dataset_path = tmp_path / "dataset.csv"
    dataframe.to_csv(dataset_path, index=False)

    config = make_dataset_config(
        path=dataset_path,
        drop_columns=["remove_me"],
    )

    loaded = load_tabular_dataset(config)

    assert list(loaded.columns) == [
        "keep_1",
        "keep_2",
    ]

    assert "remove_me" not in loaded.columns


def test_load_dataset_selects_requested_continuous_columns(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "first": [1.0, 2.0, 3.0],
            "second": [4.0, 5.0, 6.0],
            "third": [7.0, 8.0, 9.0],
        }
    )

    dataset_path = tmp_path / "dataset.csv"
    dataframe.to_csv(dataset_path, index=False)

    config = make_dataset_config(
        path=dataset_path,
        continuous_columns=[
            "third",
            "first",
        ],
    )

    loaded = load_tabular_dataset(config)

    assert list(loaded.columns) == [
        "third",
        "first",
    ]

    expected = dataframe[
        [
            "third",
            "first",
        ]
    ]

    pd.testing.assert_frame_equal(
        loaded,
        expected,
        check_dtype=False,
    )


def test_drop_columns_is_applied_before_continuous_column_selection(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0],
            "c": [7.0, 8.0, 9.0],
        }
    )

    dataset_path = tmp_path / "dataset.csv"
    dataframe.to_csv(dataset_path, index=False)

    config = make_dataset_config(
        path=dataset_path,
        drop_columns=["c"],
        continuous_columns=[
            "b",
            "a",
        ],
    )

    loaded = load_tabular_dataset(config)

    assert list(loaded.columns) == [
        "b",
        "a",
    ]


def test_load_dataset_rejects_non_numeric_selected_column(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "numeric": [1.0, 2.0, 3.0],
            "text": ["a", "b", "c"],
        }
    )

    dataset_path = tmp_path / "dataset.csv"
    dataframe.to_csv(dataset_path, index=False)

    config = make_dataset_config(
        path=dataset_path,
        continuous_columns="all",
        missing_policy="error",
    )

    with pytest.raises(
        ValueError,
        match="Non-numeric",
    ):
        load_tabular_dataset(config)


def test_non_numeric_column_can_be_excluded_before_validation(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "numeric": [1.0, 2.0, 3.0],
            "text": ["a", "b", "c"],
        }
    )

    dataset_path = tmp_path / "dataset.csv"
    dataframe.to_csv(dataset_path, index=False)

    config = make_dataset_config(
        path=dataset_path,
        continuous_columns=["numeric"],
    )

    loaded = load_tabular_dataset(config)

    assert list(loaded.columns) == ["numeric"]
    assert pd.api.types.is_numeric_dtype(
        loaded["numeric"]
    )


# ---------------------------------------------------------------------
# load_tabular_dataset: missing-value policies
# ---------------------------------------------------------------------


def test_missing_policy_error_rejects_missing_values(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0],
            "b": [4.0, 5.0, 6.0],
        }
    )

    dataset_path = tmp_path / "dataset.csv"
    dataframe.to_csv(dataset_path, index=False)

    config = make_dataset_config(
        path=dataset_path,
        missing_policy="error",
    )

    with pytest.raises(
        ValueError,
        match="Missing values",
    ):
        load_tabular_dataset(config)


def test_missing_policy_drop_rows_removes_incomplete_rows(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0, 4.0],
            "b": [5.0, 6.0, np.nan, 8.0],
        }
    )

    dataset_path = tmp_path / "dataset.csv"
    dataframe.to_csv(dataset_path, index=False)

    config = make_dataset_config(
        path=dataset_path,
        missing_policy="drop_rows",
    )

    loaded = load_tabular_dataset(config)

    expected = pd.DataFrame(
        {
            "a": [1.0, 4.0],
            "b": [5.0, 8.0],
        },
        index=[0, 3],
    )

    pd.testing.assert_frame_equal(
        loaded,
        expected,
        check_dtype=False,
    )

    assert not loaded.isna().any().any()


def test_missing_policy_drop_rows_rejects_empty_result(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "a": [np.nan, np.nan],
            "b": [1.0, np.nan],
        }
    )

    dataset_path = tmp_path / "dataset.csv"
    dataframe.to_csv(dataset_path, index=False)

    config = make_dataset_config(
        path=dataset_path,
        missing_policy="drop_rows",
    )

    with pytest.raises(
        ValueError,
        match="empty",
    ):
        load_tabular_dataset(config)


def test_load_dataset_rejects_unsupported_missing_policy(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.csv"

    pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
        }
    ).to_csv(dataset_path, index=False)

    config = make_dataset_config(
        path=dataset_path,
        missing_policy="mean_imputation",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported missing_policy",
    ):
        load_tabular_dataset(config)


def test_default_missing_policy_is_error(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0],
        }
    )

    dataset_path = tmp_path / "dataset.csv"
    dataframe.to_csv(dataset_path, index=False)

    config = make_dataset_config(
        path=dataset_path,
    )

    del config["data"]["missing_policy"]

    with pytest.raises(
        ValueError,
        match="Missing values",
    ):
        load_tabular_dataset(config)


# ---------------------------------------------------------------------
# make_dataset_summary
# ---------------------------------------------------------------------


def test_dataset_summary_has_expected_columns() -> None:
    dataframe = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [10.0, 20.0, 30.0, 40.0],
        }
    )

    summary = make_dataset_summary(dataframe)

    assert list(summary.columns) == [
        "column",
        "dtype",
        "n_rows",
        "n_missing",
        "min",
        "q05",
        "median",
        "mean",
        "q95",
        "max",
        "std",
    ]


def test_dataset_summary_has_one_row_per_variable() -> None:
    dataframe = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0],
            "c": [7.0, 8.0, 9.0],
        }
    )

    summary = make_dataset_summary(dataframe)

    assert summary.shape[0] == 3

    assert summary["column"].tolist() == [
        "a",
        "b",
        "c",
    ]


def test_dataset_summary_reports_basic_statistics_correctly() -> None:
    dataframe = pd.DataFrame(
        {
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )

    summary = make_dataset_summary(dataframe)

    row = summary.iloc[0]

    assert row["column"] == "value"
    assert row["n_rows"] == 4
    assert row["n_missing"] == 0
    assert row["min"] == pytest.approx(1.0)
    assert row["median"] == pytest.approx(2.5)
    assert row["mean"] == pytest.approx(2.5)
    assert row["max"] == pytest.approx(4.0)

    assert row["q05"] == pytest.approx(
        dataframe["value"].quantile(0.05)
    )

    assert row["q95"] == pytest.approx(
        dataframe["value"].quantile(0.95)
    )

    assert row["std"] == pytest.approx(
        dataframe["value"].std()
    )


def test_dataset_summary_reports_missing_counts() -> None:
    dataframe = pd.DataFrame(
        {
            "complete": [1.0, 2.0, 3.0],
            "incomplete": [4.0, np.nan, 6.0],
        }
    )

    summary = make_dataset_summary(dataframe)

    missing_counts = dict(
        zip(
            summary["column"],
            summary["n_missing"],
        )
    )

    assert missing_counts["complete"] == 0
    assert missing_counts["incomplete"] == 1


# ---------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------


def valid_config_payload(
    dataset_path: str = "data/example.csv",
) -> dict:
    return {
        "dataset_id": "example",
        "path": dataset_path,
        "file_type": "csv",
        "data": {
            "continuous_columns": "all",
            "missing_policy": "error",
            "drop_columns": [],
            "clip_eps": 1e-6,
        },
        "copula": {
            "selection_criterion": "bic",
            "allow_rotations": True,
            "num_threads": 1,
            "families": [
                "indep",
                "gaussian",
            ],
        },
        "outputs": {
            "artifact_dir": "artifacts/example",
            "table_dir": "reports/example",
        },
    }


def write_yaml(
    payload: dict | None,
    path: Path,
) -> None:
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            payload,
            file,
            sort_keys=False,
        )


def test_load_config_reads_valid_yaml(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"

    expected = valid_config_payload()

    write_yaml(
        expected,
        config_path,
    )

    loaded = load_config(config_path)

    assert loaded == expected


def test_load_config_accepts_string_path(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"

    expected = valid_config_payload()

    write_yaml(
        expected,
        config_path,
    )

    loaded = load_config(str(config_path))

    assert loaded == expected


def test_load_config_rejects_missing_file(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "missing.yaml"

    with pytest.raises(
        FileNotFoundError,
        match="Config file not found",
    ):
        load_config(config_path)


def test_load_config_rejects_empty_file(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "empty.yaml"
    config_path.write_text(
        "",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="empty",
    ):
        load_config(config_path)


@pytest.mark.parametrize(
    "missing_key",
    [
        "dataset_id",
        "path",
        "file_type",
        "data",
        "copula",
        "outputs",
    ],
)
def test_load_config_rejects_each_missing_required_key(
    tmp_path: Path,
    missing_key: str,
) -> None:
    config_path = (
        tmp_path
        / f"missing_{missing_key}.yaml"
    )

    payload = valid_config_payload()
    del payload[missing_key]

    write_yaml(
        payload,
        config_path,
    )

    with pytest.raises(
        ValueError,
        match="Missing required config keys",
    ) as error:
        load_config(config_path)

    assert missing_key in str(error.value)


def test_load_config_does_not_require_optional_nested_keys(
    tmp_path: Path,
) -> None:
    """
    load_config() currently validates only the six top-level keys.
    Nested validation is handled later by the consuming functions.
    """
    config_path = tmp_path / "minimal.yaml"

    payload = {
        "dataset_id": "minimal",
        "path": "data/minimal.csv",
        "file_type": "csv",
        "data": {},
        "copula": {},
        "outputs": {},
    }

    write_yaml(
        payload,
        config_path,
    )

    loaded = load_config(config_path)

    assert loaded == payload