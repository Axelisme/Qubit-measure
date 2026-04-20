from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Optional

from zcu_tools.program import TrackerProtocol


class KMeansTracker(TrackerProtocol):
    """
    Input shape: (..., m, 2) where:
        - ... : leading dimensions (treated as independent channels)
        - m   : number of samples per update
        - 2   : IQ dimensions (I, Q)
    """

    BATCH_SIZE = 256

    def __init__(
        self, leader_radius_factor: float = 1.0, max_leaders: int = 32
    ) -> None:
        self.n = 0
        self.mean: Optional[NDArray[np.float64]] = None
        self.M2: Optional[NDArray[np.float64]] = None
        self.leader_radius_factor = leader_radius_factor
        self.max_leaders = max_leaders
        # per-channel leaders, indexed by flat index into leading dims
        self._leaders: Optional[list[tuple[NDArray[np.float64], NDArray[np.int64]]]] = (
            None
        )
        self._leading_shape: tuple[int, ...] = ()

    @staticmethod
    def _merge(
        n1: int,
        mean1: NDArray[np.float64],
        M2_1: NDArray[np.float64],
        n2: int,
        mean2: NDArray[np.float64],
        M2_2: NDArray[np.float64],
    ) -> tuple[int, NDArray[np.float64], NDArray[np.float64]]:
        """Merge two sets of statistics. Supports leading dimensions."""
        n = n1 + n2
        mean = (mean1 * n1 + mean2 * n2) / float(n)

        delta = mean1 - mean2  # (..., 2)
        delta_outer = delta[..., :, None] * delta[..., None, :]
        M2 = M2_1 + M2_2 + delta_outer * (n1 * n2 / n)

        return n, mean, M2

    def _update_one_channel(
        self,
        centers: NDArray[np.float64],
        counts: NDArray[np.int64],
        points: NDArray[np.float64],
        threshold: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Leader algorithm: assign points to existing leaders within
        ``threshold``; spawn new leaders for the rest."""
        # Phase 1: vectorized assignment to existing leaders
        if len(centers) > 0:
            dists = np.linalg.norm(
                points[:, None, :] - centers[None, :, :], axis=-1
            )  # (m, K)
            min_idx = np.argmin(dists, axis=1)  # (m,)
            min_dist = dists[np.arange(len(points)), min_idx]
            matched = min_dist < threshold

            new_centers = centers.astype(np.float64, copy=True)
            new_counts = counts.astype(np.int64, copy=True)
            for k in range(len(centers)):
                mask = matched & (min_idx == k)
                n_new = int(mask.sum())
                if n_new == 0:
                    continue
                sum_new = points[mask].sum(axis=0)
                n_old = int(new_counts[k])
                new_centers[k] = (new_centers[k] * n_old + sum_new) / (n_old + n_new)
                new_counts[k] = n_old + n_new

            unmatched = points[~matched]
        else:
            new_centers = np.empty((0, 2), dtype=np.float64)
            new_counts = np.empty((0,), dtype=np.int64)
            unmatched = points

        # Phase 2: sequential leader algorithm on unmatched points
        extra_centers: list[NDArray[np.float64]] = []
        extra_counts: list[int] = []
        for p in unmatched:
            if not extra_centers:
                extra_centers.append(p.astype(np.float64, copy=True))
                extra_counts.append(1)
                continue
            arr = np.asarray(extra_centers)
            d = np.linalg.norm(arr - p, axis=1)
            i = int(np.argmin(d))
            if d[i] < threshold:
                n_old = extra_counts[i]
                extra_centers[i] = (extra_centers[i] * n_old + p) / (n_old + 1)
                extra_counts[i] = n_old + 1
            else:
                extra_centers.append(p.astype(np.float64, copy=True))
                extra_counts.append(1)

        if extra_centers:
            new_centers = np.vstack([new_centers, np.asarray(extra_centers)])
            new_counts = np.concatenate(
                [new_counts, np.asarray(extra_counts, dtype=np.int64)]
            )

        # Prune to top max_leaders by count (preserve memory)
        if len(new_counts) > self.max_leaders:
            order = np.argsort(new_counts)[::-1][: self.max_leaders]
            new_centers = new_centers[order]
            new_counts = new_counts[order]

        return new_centers, new_counts

    def _update_chunk(self, points: NDArray[np.float64]) -> None:
        """
        Update statistics with new points.

        Args:
            points: shape (..., m, 2) where m is number of samples
        """
        assert len(points.shape) >= 2
        assert points.shape[-1] == 2, "Last dimension must be 2 (I, Q)"

        cur_n = points.shape[-2]
        assert cur_n >= 1

        points_mean = np.mean(points, axis=-2)  # (..., 2)
        centered_points = points - points_mean[..., None, :]  # (..., m, 2)
        cur_M2 = np.einsum("...mi,...mj->...ij", centered_points, centered_points)

        # update mean / M2 first so threshold uses post-merge cov
        if self.n == 0:
            self.n = cur_n
            self.mean = points_mean
            self.M2 = cur_M2
            self._leading_shape = points.shape[:-2]
        else:
            assert self.mean is not None
            assert self.M2 is not None
            self.n, self.mean, self.M2 = self._merge(
                self.n, self.mean, self.M2, cur_n, points_mean, cur_M2
            )

        # leader-algorithm update for mode estimation
        flat_points = points.reshape(-1, cur_n, 2)
        C = flat_points.shape[0]

        if self._leaders is None:
            self._leaders = [
                (np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.int64))
                for _ in range(C)
            ]

        # threshold per channel from running covariance trace
        cov = self.covariance  # (..., 2, 2)
        trace_half = (cov[..., 0, 0] + cov[..., 1, 1]) / 2.0  # (...,)
        sigma = np.sqrt(np.clip(trace_half, 1e-24, None))
        # fall back to within-batch std if running cov hasn't stabilized
        flat_sigma = np.broadcast_to(sigma, self._leading_shape).reshape(-1)
        thresholds = flat_sigma * self.leader_radius_factor

        for c in range(C):
            centers, counts = self._leaders[c]
            self._leaders[c] = self._update_one_channel(
                centers, counts, flat_points[c], float(thresholds[c])
            )

    def update(self, points: NDArray[np.float64]) -> None:
        """Update statistics; process at most ``BATCH_SIZE`` points per chunk."""
        assert len(points.shape) >= 2
        assert points.shape[-1] == 2, "Last dimension must be 2 (I, Q)"
        total_n = points.shape[-2]
        assert total_n >= 1

        for start in range(0, total_n, self.BATCH_SIZE):
            end = min(start + self.BATCH_SIZE, total_n)
            self._update_chunk(points[..., start:end, :])

    @property
    def covariance(self) -> NDArray[np.float64]:
        """Returns covariance matrix with shape (..., 2, 2)."""
        if self.M2 is None:
            raise ValueError("Covariance is not available yet")
        if self.n == 1:
            return self.M2  # (..., 2, 2)

        return self.M2 / (self.n - 1)

    @property
    def leader_center(self) -> NDArray[np.float64]:
        """Mode estimate (highest-count leader) per channel; shape (..., 2)."""
        if self._leaders is None or not self._leaders:
            raise ValueError("leader_center is not available yet")

        modes = np.empty((len(self._leaders), 2), dtype=np.float64)
        for c, (centers, counts) in enumerate(self._leaders):
            if len(counts) == 0:
                modes[c] = np.nan
            else:
                modes[c] = centers[int(np.argmax(counts))]
        return modes.reshape(self._leading_shape + (2,))
