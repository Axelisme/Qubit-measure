from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from zcu_tools.program import TrackerProtocol


class _CentralMoments(NamedTuple):
    """Keep each sample group's statistics together between chunking and merging.

    Arrays share the same leading channel dimensions. Construction retains their
    references without copying; only the tuple's field bindings are immutable.
    """

    n: int
    """Sample count shared by all leading channels."""
    mean: NDArray[np.float64]
    """Mean IQ vector, shape ``(..., 2)``; array reference is not copied."""
    M2: NDArray[np.float64]
    """Second central sum, shape ``(..., 2, 2)``; array reference is not copied."""
    M3: NDArray[np.float64]
    """Third central sum, shape ``(..., 2, 2, 2)``; array reference is not copied."""


def _merge_moments3(first: _CentralMoments, second: _CentralMoments) -> _CentralMoments:
    """Combine two nonempty groups with matching leading channel shapes.

    Return merged statistics in newly computed arrays, leaving inputs unchanged.

    For 1D (scalar case):
        M3 = M3_A + M3_B
             + 3·δ·(n_A·M2_B − n_B·M2_A)/n
             + δ³·n_A·n_B·(n_A − n_B)/n²
    Tensor generalization keeps full symmetry in (i, j, k).
    """
    n1, mean1, M2_1, M3_1 = first
    n2, mean2, M2_2, M3_2 = second
    n = n1 + n2
    mean = (mean1 * n1 + mean2 * n2) / float(n)
    delta = mean2 - mean1  # (..., 2)
    d_i = delta[..., :, None, None]
    d_j = delta[..., None, :, None]
    d_k = delta[..., None, None, :]

    delta_outer = delta[..., :, None] * delta[..., None, :]
    M2 = M2_1 + M2_2 + delta_outer * (n1 * n2 / n)

    # 3rd moment combine term
    T = (n1 * M2_2 - n2 * M2_1) / float(n)  # (..., 2, 2)
    # 3·sym(δ ⊗ T)_ijk = δ_i T_jk + δ_j T_ik + δ_k T_ij
    sym_term = (
        d_i * T[..., None, :, :] + d_j * T[..., :, None, :] + d_k * T[..., :, :, None]
    )
    ddd = d_i * d_j * d_k
    cubic = ddd * (n1 * n2 * (n1 - n2) / float(n * n))
    M3 = M3_1 + M3_2 + sym_term + cubic
    return _CentralMoments(n, mean, M2, M3)


class MomentTracker(TrackerProtocol):
    """Incremental 1st/2nd/3rd central moment tracker for 2D IQ data.

    Input shape: ``(..., m, 2)`` — leading dims are independent channels,
    ``m`` is samples per update, last dim is (I, Q). State shapes:
        n:     int (shared over leading dims; all channels get same count)
        mean:  (..., 2)
        M2:    (..., 2, 2)
        M3:    (..., 2, 2, 2)
    """

    BATCH_SIZE = 256

    def __init__(self) -> None:
        self.n = 0
        self._mean: NDArray[np.float64] | None = None
        self.M2: NDArray[np.float64] | None = None
        self.M3: NDArray[np.float64] | None = None
        self._leading_shape: tuple[int, ...] = ()

    def _chunk_moments(self, points: NDArray[np.float64]) -> _CentralMoments:
        n = points.shape[-2]
        mean = np.mean(points, axis=-2)
        centered = points - mean[..., None, :]
        M2 = np.einsum("...mi,...mj->...ij", centered, centered)
        M3 = np.einsum("...mi,...mj,...mk->...ijk", centered, centered, centered)
        return _CentralMoments(n, mean.astype(np.float64, copy=False), M2, M3)

    def update(self, points: NDArray[np.float64]) -> None:
        assert points.ndim >= 2
        assert points.shape[-1] == 2, "Last dimension must be 2 (I, Q)"
        total = points.shape[-2]
        assert total >= 1

        leading_shape = points.shape[:-2]
        if self.n == 0:
            self._leading_shape = leading_shape
        elif leading_shape != self._leading_shape:
            raise ValueError(
                f"Leading shape mismatch: expected {self._leading_shape}, got {leading_shape}"
            )

        for start in range(0, total, self.BATCH_SIZE):
            end = min(start + self.BATCH_SIZE, total)
            chunk = points[..., start:end, :]
            batch = self._chunk_moments(chunk)
            n_b, mean_b, M2_b, M3_b = batch

            if self.n == 0:
                self.n = n_b
                self._mean = mean_b.astype(np.float64, copy=True)
                self.M2 = M2_b.astype(np.float64, copy=True)
                self.M3 = M3_b.astype(np.float64, copy=True)
            else:
                assert self._mean is not None
                assert self.M2 is not None
                assert self.M3 is not None
                self.n, self._mean, self.M2, self.M3 = _merge_moments3(
                    _CentralMoments(self.n, self._mean, self.M2, self.M3),
                    batch,
                )

    @property
    def mean(self) -> NDArray[np.float64]:
        if self._mean is None:
            raise ValueError("Mean is not available yet")
        return self._mean

    @property
    def covariance(self) -> NDArray[np.float64]:
        if self.M2 is None:
            raise ValueError("Covariance is not available yet")
        if self.n <= 1:
            return self.M2
        return self.M2 / (self.n - 1)

    @property
    def third_moment(self) -> NDArray[np.float64]:
        """Population third central moment ``E[(x-μ)⊗(x-μ)⊗(x-μ)]`` (``= M3 / n``)."""
        if self.M3 is None:
            raise ValueError("Third moment is not available yet")
        if self.n < 1:
            return np.full_like(self.M3, np.nan)
        return self.M3 / self.n
