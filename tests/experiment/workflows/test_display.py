"""Observe sparse display assembly through the public workflow interface."""

from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.experiment.workflows import assemble_rows, assemble_scalars


def test_rows_preserve_indices_missing_rows_and_supplied_nan() -> None:
    result = assemble_rows(
        [
            (2, np.array([5.0, 6.0], dtype=np.float64)),
            (0, np.array([np.nan, 2.0], dtype=np.float64)),
        ],
        n_rows=3,
        length=2,
    )
    np.testing.assert_equal(
        result.values, [[np.nan, 2.0], [np.nan, np.nan], [5.0, 6.0]]
    )
    np.testing.assert_array_equal(result.filled, [True, False, True])


def test_rows_do_not_share_writable_storage_with_the_input() -> None:
    row = np.array([1.0, 2.0], dtype=np.float64)
    result = assemble_rows([(0, row)], n_rows=1, length=2)
    row[0] = 9.0
    assert result.values[0, 0] == 1.0
    result.values[0, 1] = 8.0
    assert row[1] == 2.0


def test_empty_dimensions_return_empty_data_without_filled_rows() -> None:
    rows = assemble_rows([], n_rows=0, length=2)
    assert rows.values.shape == (0, 2)
    assert rows.filled.shape == (0,)
    points = assemble_scalars([], n=0)
    assert points.values.shape == (0,)
    assert points.filled.shape == (0,)


@pytest.mark.parametrize("index", [-1, 2])
def test_rows_reject_indices_outside_the_matrix(index: int) -> None:
    row = np.array([1.0], dtype=np.float64)
    with pytest.raises(ValueError, match="outside"):
        assemble_rows([(index, row)], n_rows=2, length=1)


def test_rows_reject_duplicate_indices() -> None:
    row = np.array([1.0], dtype=np.float64)
    with pytest.raises(ValueError, match="Duplicate row"):
        assemble_rows([(0, row), (0, row)], n_rows=1, length=1)


@pytest.mark.parametrize("shape", [(2,), (1, 1)])
def test_rows_reject_wrong_length_or_rank(shape: tuple[int, ...]) -> None:
    row = np.ones(shape, dtype=np.float64)
    with pytest.raises(ValueError, match="vector of length"):
        assemble_rows([(0, row)], n_rows=1, length=1)


@pytest.mark.parametrize("dimensions", [(-1, 1), (1, -1)])
def test_rows_reject_negative_dimensions(dimensions: tuple[int, int]) -> None:
    n_rows, length = dimensions
    with pytest.raises(ValueError, match="nonnegative"):
        assemble_rows([], n_rows=n_rows, length=length)


def test_scalars_preserve_indices_and_distinguish_missing_from_supplied_nan() -> None:
    result = assemble_scalars([(2, 4.0), (0, np.nan)], n=4)
    np.testing.assert_equal(result.values, [np.nan, np.nan, 4.0, np.nan])
    np.testing.assert_array_equal(result.filled, [True, False, True, False])


@pytest.mark.parametrize("index", [-1, 2])
def test_scalars_reject_indices_outside_the_vector(index: int) -> None:
    with pytest.raises(ValueError, match="outside"):
        assemble_scalars([(index, 1.0)], n=2)


def test_scalars_reject_duplicate_indices() -> None:
    with pytest.raises(ValueError, match="Duplicate point"):
        assemble_scalars([(0, 1.0), (0, 2.0)], n=1)


def test_scalars_reject_negative_size() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        assemble_scalars([], n=-1)
