from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Optional

from zcu_tools.program import TrackerProtocol


def _merge_moments2(
    n1: float,
    mean1: NDArray[np.float64],
    M2_1: NDArray[np.float64],
    n2: int,
    mean2: NDArray[np.float64],
    M2_2: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    if n1 <= 0:
        return float(n2), mean2.copy(), M2_2.copy()
    if n2 <= 0:
        return n1, mean1, M2_1

    n = n1 + float(n2)
    delta = mean2 - mean1
    mean = mean1 + delta * (float(n2) / n)
    delta_outer = delta[:, None] * delta[None, :]
    M2 = M2_1 + M2_2 + delta_outer * (n1 * float(n2) / n)
    return n, mean, M2


class KMeansTracker(TrackerProtocol):
    """Incremental KMeans-like tracker for 2D IQ points.

    Input shape: ``(..., m, 2)`` where ``...`` are leading sweep dimensions.
    When ``share_axis`` is set, those leading axes share one cluster model.
    """

    BATCH_SIZE = 256

    def __init__(
        self,
        max_cluster_num: int = 3,
        share_axis: Optional[int | tuple[int, ...]] = None,
    ) -> None:
        if max_cluster_num < 1:
            raise ValueError("max_cluster_num must be >= 1")
        self.max_cluster_num = int(max_cluster_num)
        self.share_axis = share_axis

        self.n = 0
        self._leading_shape: Optional[tuple[int, ...]] = None
        self._share_axis_norm: tuple[int, ...] = ()
        self._group_shape: tuple[int, ...] = ()
        self._group_axes: tuple[int, ...] = ()
        self._group_count = 0

        self._cluster_counts: Optional[NDArray[np.int32]] = None
        self._cluster_sizes: Optional[NDArray[np.float64]] = None
        self._cluster_means: Optional[NDArray[np.float64]] = None
        self._cluster_M2: Optional[NDArray[np.float64]] = None

    def _normalize_share_axis(self, leading_ndim: int) -> tuple[int, ...]:
        if self.share_axis is None:
            return ()

        axes = (
            (self.share_axis,) if isinstance(self.share_axis, int) else self.share_axis
        )
        norm_axes: list[int] = []
        for axis in axes:
            axis_i = int(axis)
            if axis_i < 0:
                axis_i += leading_ndim
            if axis_i < 0 or axis_i >= leading_ndim:
                raise ValueError(
                    f"share_axis contains out-of-range axis {axis} for leading ndim {leading_ndim}"
                )
            norm_axes.append(axis_i)
        return tuple(sorted(set(norm_axes)))

    def _init_state(self, leading_shape: tuple[int, ...]) -> None:
        leading_ndim = len(leading_shape)
        self._share_axis_norm = self._normalize_share_axis(leading_ndim)
        self._group_axes = tuple(
            i for i in range(leading_ndim) if i not in self._share_axis_norm
        )
        self._group_shape = tuple(leading_shape[i] for i in self._group_axes)
        self._group_count = int(np.prod(self._group_shape)) if self._group_shape else 1

        g = self._group_count
        k = self.max_cluster_num
        self._cluster_counts = np.zeros((g,), dtype=np.int32)
        self._cluster_sizes = np.zeros((g, k), dtype=np.float64)
        self._cluster_means = np.zeros((g, k, 2), dtype=np.float64)
        self._cluster_M2 = np.zeros((g, k, 2, 2), dtype=np.float64)

    def _iter_chunks(self, points: NDArray[np.float64]) -> list[NDArray[np.float64]]:
        total = points.shape[-2]
        return [
            cast_chunk.astype(np.float64, copy=False)
            for cast_chunk in (
                points[..., start : min(start + self.BATCH_SIZE, total), :]
                for start in range(0, total, self.BATCH_SIZE)
            )
        ]

    @staticmethod
    def _batch_moments(
        points: NDArray[np.float64],
    ) -> tuple[int, NDArray[np.float64], NDArray[np.float64]]:
        n = points.shape[0]
        mean = np.mean(points, axis=0)
        centered = points - mean[None, :]
        M2 = centered.T @ centered
        return n, mean.astype(np.float64, copy=False), M2.astype(np.float64, copy=False)

    @staticmethod
    def _select_seed_centers(
        points: NDArray[np.float64], num_centers: int
    ) -> NDArray[np.float64]:
        if num_centers <= 0:
            return np.zeros((0, 2), dtype=np.float64)
        centers = np.empty((num_centers, 2), dtype=np.float64)
        centers[0] = points[0]
        min_dist2 = np.sum((points - centers[0]) ** 2, axis=1)
        for i in range(1, num_centers):
            idx = int(np.argmax(min_dist2))
            centers[i] = points[idx]
            min_dist2 = np.minimum(
                min_dist2, np.sum((points - centers[i]) ** 2, axis=1)
            )
        return centers

    def _expand_clusters(self, g: int, points: NDArray[np.float64]) -> None:
        assert self._cluster_counts is not None
        assert self._cluster_means is not None
        active = int(self._cluster_counts[g])
        if points.shape[0] == 0:
            return

        if active == 0:
            n_init = min(self.max_cluster_num, points.shape[0])
            self._cluster_means[g, :n_init, :] = self._select_seed_centers(
                points, n_init
            )
            self._cluster_counts[g] = n_init
            active = n_init

        while active < self.max_cluster_num:
            dist2 = np.sum(
                (points[:, None, :] - self._cluster_means[g, :active, :][None, :, :])
                ** 2,
                axis=-1,
            )
            min_dist2 = np.min(dist2, axis=1)
            farthest = int(np.argmax(min_dist2))
            if min_dist2[farthest] <= 0:
                break
            self._cluster_means[g, active, :] = points[farthest]
            active += 1
            self._cluster_counts[g] = active

    def _update_group(self, g: int, points: NDArray[np.float64]) -> None:
        assert self._cluster_counts is not None
        assert self._cluster_sizes is not None
        assert self._cluster_means is not None
        assert self._cluster_M2 is not None
        if points.shape[0] == 0:
            return

        self._expand_clusters(g, points)
        active = int(self._cluster_counts[g])
        if active <= 0:
            return

        dist2 = np.sum(
            (points[:, None, :] - self._cluster_means[g, :active, :][None, :, :]) ** 2,
            axis=-1,
        )
        labels = np.argmin(dist2, axis=1)

        for k in range(active):
            mask = labels == k
            if not np.any(mask):
                continue
            pts = points[mask]
            n_b, mean_b, M2_b = self._batch_moments(pts)
            n_old = self._cluster_sizes[g, k]
            mean_old = self._cluster_means[g, k]
            M2_old = self._cluster_M2[g, k]
            n_new, mean_new, M2_new = _merge_moments2(
                n_old, mean_old, M2_old, n_b, mean_b, M2_b
            )
            self._cluster_sizes[g, k] = n_new
            self._cluster_means[g, k] = mean_new
            self._cluster_M2[g, k] = M2_new

    def update(self, points: NDArray[np.float64]) -> None:
        assert points.ndim >= 2
        assert points.shape[-1] == 2, "Last dimension must be 2 (I, Q)"
        total = points.shape[-2]
        assert total >= 1

        leading_shape = points.shape[:-2]
        if self._leading_shape is None:
            self._leading_shape = leading_shape
            self._init_state(leading_shape)
        elif leading_shape != self._leading_shape:
            raise ValueError(
                f"Leading shape mismatch: expected {self._leading_shape}, got {leading_shape}"
            )

        leading_ndim = len(leading_shape)
        shared_axes = tuple(
            i for i in range(leading_ndim) if i in self._share_axis_norm
        )
        permute = (*self._group_axes, *shared_axes, leading_ndim, leading_ndim + 1)

        for chunk in self._iter_chunks(points):
            reshaped = np.transpose(chunk, permute)
            group_shape = tuple(leading_shape[i] for i in self._group_axes)
            shared_shape = tuple(leading_shape[i] for i in shared_axes)
            reshaped = reshaped.reshape(
                int(np.prod(group_shape)) if group_shape else 1,
                int(np.prod(shared_shape)) * reshaped.shape[-2]
                if shared_shape
                else reshaped.shape[-2],
                2,
            )
            for g in range(reshaped.shape[0]):
                self._update_group(g, reshaped[g])
            self.n += chunk.shape[-2]

    def _require_state(
        self,
    ) -> tuple[
        NDArray[np.int32], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
    ]:
        if (
            self._cluster_counts is None
            or self._cluster_sizes is None
            or self._cluster_means is None
            or self._cluster_M2 is None
        ):
            raise ValueError("Cluster statistics are not available yet")
        return (
            self._cluster_counts,
            self._cluster_sizes,
            self._cluster_means,
            self._cluster_M2,
        )

    def _reshape_count(self, arr: NDArray[np.int32]) -> NDArray[np.int32]:
        return arr.reshape((*self._group_shape, *arr.shape[1:]))

    def _reshape_float(self, arr: NDArray[np.float64]) -> NDArray[np.float64]:
        return arr.reshape((*self._group_shape, *arr.shape[1:]))

    @property
    def cluster_count(self) -> NDArray[np.int32]:
        counts, _, _, _ = self._require_state()
        return self._reshape_count(counts)

    @property
    def cluster_weight(self) -> NDArray[np.float64]:
        _, sizes, _, _ = self._require_state()
        denom = np.sum(sizes, axis=1, keepdims=True)
        weight = np.zeros_like(sizes, dtype=np.float64)
        np.divide(sizes, denom, where=denom > 0, out=weight)
        return self._reshape_float(weight.astype(np.float64, copy=False))

    @property
    def cluster_mean(self) -> NDArray[np.float64]:
        counts, _, means, _ = self._require_state()
        out = means.copy()
        inactive = np.arange(self.max_cluster_num)[None, :] >= counts[:, None]
        out[inactive] = np.nan
        return self._reshape_float(out.astype(np.float64, copy=False))

    @property
    def cluster_center(self) -> NDArray[np.complex128]:
        mean = self.cluster_mean
        center = mean[..., 0] + 1j * mean[..., 1]
        return center.astype(np.complex128, copy=False)

    @property
    def cluster_covariance(self) -> NDArray[np.float64]:
        counts, sizes, _, M2 = self._require_state()
        out = np.full_like(M2, np.nan, dtype=np.float64)
        for k in range(self.max_cluster_num):
            n_k = sizes[:, k]
            valid = n_k > 0
            if not np.any(valid):
                continue
            denom = np.where(n_k > 1, n_k - 1.0, n_k)
            out[valid, k] = M2[valid, k] / denom[valid, None, None]

        inactive = np.arange(self.max_cluster_num)[None, :] >= counts[:, None]
        out[inactive] = np.nan
        return self._reshape_float(out.astype(np.float64, copy=False))
