"""Evolve-local calibrated interpolation of physical arrays, never eigenvectors."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from threading import RLock

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class _Tile:
    lower: float
    upper: float
    # None marks an unprofitable or unconverged tile: build each carrier exactly.
    coefficients: NDArray[np.complex128] | None


class CarrierWorkspace:
    """Thread-safe physical-block carrier interpolation for one evolve.

    frequencies is the finite signed GHz scan, including every planned carrier.
    groups optionally supplies aligned hashable noncarrier physical-input keys.
    The Sambe caller groups by model identity/level, port, peak bound and
    subdivisions; absent groups means one group (). The cost rule applies to
    each group separately, before any construction.
    Interpolation is enabled only for >9 distinct, nonzero, same-sign carriers
    with max(abs(f))/min(abs(f)) <= 1.1. Nine is the build cost: six degree-five
    Chebyshev-Gauss nodes plus three independent probes. Wider scans use exact
    builds, avoiding low-frequency phase turns and retained failed-parent data.
    This conservative policy admits at most one 1.1-ratio tile, clipped to the
    scan endpoints. Refinement bisects that tile, at most sixteen levels.

    Each probe (-0.83, 0, 0.61 in normalized tile coordinates) must have a sum of
    trailing-matrix Frobenius differences <=1e-7. Failed probes subdivide only
    where >9 planned carriers remain; other children and depth-limited tiles
    use exact builds. This is calibrated, not a rigorous interval bound.
    Callback errors propagate. No density, eigenvector gauge, or global state
    enters the workspace. The Engine owns it only until this evolve finishes.
    nbytes charges all retained numeric arrays as live prepared data.
    """

    def __init__(
        self,
        frequencies: NDArray[np.float64],
        *,
        groups: Sequence[tuple[object, ...]] | None = None,
    ) -> None:
        if frequencies.ndim != 1 or not np.all(np.isfinite(frequencies)):
            raise ValueError("carriers must be a finite one-dimensional GHz scan")
        if groups is None:
            groups = [()] * len(frequencies)
        if len(groups) != len(frequencies):
            raise ValueError("groups must align with carriers")
        planned: dict[tuple[object, ...], list[float]] = {}
        for group, frequency in zip(groups, frequencies, strict=True):
            planned.setdefault(group, []).append(float(frequency))
        self._grids: dict[tuple[object, ...], NDArray[np.float64]] = {}
        for group, carriers in planned.items():
            grid = np.unique(carriers)
            magnitudes = np.abs(grid)
            if (
                len(grid) > 9
                and np.all(grid * grid[0] > 0)
                and magnitudes.max() <= 1.1 * magnitudes.min()
            ):
                self._grids[group] = grid
        self._tiles: dict[
            tuple[tuple[object, ...], tuple[object, ...]], list[_Tile]
        ] = {}
        self._lock = RLock()

    @property
    def nbytes(self) -> int:
        """Retained carrier-grid and polynomial bytes, not transient build scratch."""
        with self._lock:
            return sum(grid.nbytes for grid in self._grids.values()) + sum(
                tile.coefficients.nbytes
                for tiles in self._tiles.values()
                for tile in tiles
                if tile.coefficients is not None
            )

    def evaluate(
        self,
        identity: tuple[object, ...],
        frequency: float,
        build: Callable[[float], NDArray[np.complex128]],
        *,
        group: tuple[object, ...] = (),
    ) -> NDArray[np.complex128]:
        """Return an owned array at this exact signed carrier, not a nearest node.

        group selects a preplanned physical-input grid, default (). Unknown or
        unprofitable groups use exact builds. identity is a hashable family key
        fixing every physical input except
        carrier: model identity/level, port, peak gain, subdivision count, harmonic
        cutoff and amplitude nodes, and whether arrays are peak pairs or amplitude
        coefficients. A workspace cannot cross source or budget boundaries.
        build(f) must independently return finite complex physical arrays with
        identical shape (..., M, N) for every carrier in the scan, without mutating
        prior results. The callback may run at off-grid nodes and probes. Its
        exceptions propagate; they do not trigger numerical rerouting. Outside
        the scan, or for unprofitable tiles, return build(frequency) unchanged.
        Concurrent requests serialize tile construction; returned arrays never
        alias retained coefficients. Nonfinite frequency raises ValueError.
        """
        if not np.isfinite(frequency):
            raise ValueError("carrier must be finite")
        frequencies = self._grids.get(group)
        if frequencies is None or not frequencies[0] <= frequency <= frequencies[-1]:
            return build(frequency)
        with self._lock:
            tile_key = (group, identity)
            tiles = self._tiles.get(tile_key)
            if tiles is None:
                tiles = []
                self._refine(
                    frequencies, frequencies[0], frequencies[-1], build, tiles, 0
                )
                self._tiles[tile_key] = tiles
            for tile in tiles:
                if tile.lower <= frequency <= tile.upper:
                    return (
                        build(frequency)
                        if tile.coefficients is None
                        else self._value(tile, frequency)
                    )
        raise RuntimeError("carrier tile did not cover the requested frequency")

    @staticmethod
    def _value(tile: _Tile, frequency: float) -> NDArray[np.complex128]:
        if tile.coefficients is None:
            raise ValueError("exact tile has no polynomial")
        x = (2 * frequency - tile.lower - tile.upper) / (tile.upper - tile.lower)
        return np.asarray(
            np.polynomial.chebyshev.chebval(x, tile.coefficients), dtype=np.complex128
        )

    def _refine(
        self,
        frequencies: NDArray[np.float64],
        lower: float,
        upper: float,
        build: Callable[[float], NDArray[np.complex128]],
        tiles: list[_Tile],
        depth: int,
    ) -> None:
        count = np.searchsorted(frequencies, upper, side="right") - np.searchsorted(
            frequencies, lower
        )
        if count <= 9 or depth >= 16:
            tiles.append(_Tile(lower, upper, None))
            return
        nodes = np.cos(np.pi * (np.arange(6) + 0.5) / 6)
        values = np.array(
            [build((lower + upper + x * (upper - lower)) / 2) for x in nodes]
        )
        transform = 2 / 6 * np.cos(np.arange(6)[:, None] * np.arccos(nodes))
        transform[0] /= 2
        coefficients = np.tensordot(transform, values, axes=(1, 0))
        tile = _Tile(lower, upper, coefficients)
        error = 0.0
        for x in (-0.83, 0.0, 0.61):
            frequency = (lower + upper + x * (upper - lower)) / 2
            difference = self._value(tile, frequency) - build(frequency)
            error = max(error, float(np.sum(np.linalg.norm(difference, axis=(-2, -1)))))
        if error <= 1e-7:
            coefficients.setflags(write=False)
            tiles.append(tile)
        else:
            middle = (lower + upper) / 2
            self._refine(frequencies, lower, middle, build, tiles, depth + 1)
            self._refine(frequencies, middle, upper, build, tiles, depth + 1)
