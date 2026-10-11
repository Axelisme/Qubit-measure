"""Detached sparse-to-dense workflow display data."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Dense:
    """Detached display arrays made by assemble_rows/assemble_scalars.

    ``values`` is float64, with NaN at missing positions. It is a matrix for
    rows or a vector for scalars. ``filled`` is a bool vector marking supplied
    rows/positions, even when the supplied value itself is NaN.
    """

    values: NDArray[np.float64]
    filled: NDArray[np.bool_]


def assemble_rows(
    rows: Sequence[tuple[int, NDArray[np.float64]]], *, n_rows: int, length: int
) -> Dense:
    """Place indexed rows into a detached (n_rows, length) float64 matrix.

    Dimensions may be zero, never negative. Duplicate/out-of-range indices and
    non-vector or wrong-length rows raise ValueError. No interpolation occurs.
    """
    if n_rows < 0 or length < 0:
        raise ValueError("Display dimensions must be nonnegative")
    values = np.full((n_rows, length), np.nan, dtype=np.float64)
    filled = np.zeros(n_rows, dtype=np.bool_)
    for index, row in rows:
        if not 0 <= index < n_rows:
            raise ValueError(f"Row index {index} is outside {n_rows} rows")
        if filled[index]:
            raise ValueError(f"Duplicate row index {index}")
        if row.ndim != 1 or row.size != length:
            raise ValueError(f"Row {index} must be a vector of length {length}")
        values[index] = row
        filled[index] = True
    return Dense(values, filled)


def assemble_scalars(points: Sequence[tuple[int, float]], *, n: int) -> Dense:
    """Place indexed values into a detached length-n float64 vector.

    n may be zero, never negative. Duplicate/out-of-range indices raise
    ValueError. Missing points remain NaN with filled=False.
    """
    if n < 0:
        raise ValueError("Display size must be nonnegative")
    values = np.full(n, np.nan, dtype=np.float64)
    filled = np.zeros(n, dtype=np.bool_)
    for index, value in points:
        if not 0 <= index < n:
            raise ValueError(f"Point index {index} is outside {n} points")
        if filled[index]:
            raise ValueError(f"Duplicate point index {index}")
        values[index] = value
        filled[index] = True
    return Dense(values, filled)
