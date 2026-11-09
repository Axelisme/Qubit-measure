"""Independent scipy.expm rate truths and shot generation for the T1 seam."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm
from zcu_tools.analysis.readout_chain import CycleModel, CycleNuisances, CycleSums

from ._cycle_samples import cycle_samples

DELAYS = np.linspace(0.0, 100.0, 13)
THERMAL = np.array([0.84, 0.10, 0.06])
PI_SWAP = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
_FLAT = CycleModel.identical(
    lambda x, p: np.zeros_like(x),
    ("unused",),
    ((0.0, 1.0),),
    derivative=lambda x, p: np.zeros((len(x), 1)),
)


@dataclass(frozen=True)
class RateTruth:
    """Three pair times (01,02,12), microseconds; infinity removes a pair."""

    name: str
    pair_times: tuple[float, float, float]

    def generator(self) -> np.ndarray:
        """Build the directed generator independently of the symmetric fit formula."""
        matrix = np.zeros((3, 3))
        for (i, j), time in zip(((0, 1), (0, 2), (1, 2)), self.pair_times, strict=True):
            matrix[i, j] = THERMAL[j] / (THERMAL[i] + THERMAL[j]) / time
            matrix[j, i] = THERMAL[i] / (THERMAL[i] + THERMAL[j]) / time
        return matrix - np.diag(matrix.sum(axis=1))

    def nuisance(self, delay: float, contrast: float) -> CycleNuisances:
        """Fixed kernel at delay with the study's readout geometry and transitions."""
        kernel = PI_SWAP @ expm(self.generator() * delay)
        kernel = np.clip(kernel, 0.0, 1.0)
        kernel /= kernel.sum(axis=1, keepdims=True)
        centers = contrast * np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.4 * np.cos(1.6), 1.4 * np.sin(1.6)],
            ]
        )
        return CycleNuisances(
            np.repeat(kernel[:, None, :], 2, axis=1),
            centers,
            1.0,
            THERMAL.copy(),
            np.array([0.95, 0.8, 0.95]),
            np.array([0.63, 0.27, 0.1]),
        )

    def sums(
        self, *, contrast: float = 2.1, cycles: int = 3000, seed: int | None = None
    ) -> CycleSums:
        """Exact moments or shots, with an independent expm kernel at every delay."""
        sums, products = [], np.zeros((65, 65))
        for i, delay in enumerate(DELAYS):
            nuisance = self.nuisance(float(delay), contrast)
            if seed is None:
                moments = _FLAT.moments(np.array([delay]), {"unused": 0.0}, nuisance)
                mean = moments.means[0]
                sums.append(cycles * mean)
                products += cycles * (moments.covariance[0] + np.outer(mean, mean))
            else:
                shots = cycle_samples(
                    _FLAT,
                    float(delay),
                    {"unused": 0.0},
                    nuisance,
                    cycles,
                    seed + 1000 * i,
                )
                sums.append(shots.sum(axis=0))
                products += shots.T @ shots
        return CycleSums(
            DELAYS.copy(), np.full(len(DELAYS), float(cycles)), np.array(sums), products
        )

    def modes(self) -> tuple[np.ndarray, np.ndarray]:
        """Times and L1 amplitudes from the nonsymmetric generator eigenvectors."""
        values, right = np.linalg.eig(self.generator())
        order = np.argsort(-np.real(values))
        values, right = np.real(values[order]), np.real(right[:, order])
        left = np.linalg.inv(right)
        amplitudes = np.array(
            [
                THERMAL @ np.abs(PI_SWAP @ np.outer(right[:, k], left[k])).sum(axis=1)
                for k in (1, 2)
            ]
        )
        return -1 / values[1:], amplitudes


STAR = RateTruth("star", (20.0, 12.0, np.inf))
LADDER = RateTruth("ladder", (8.422325225180606, np.inf, 8.422325225180606))
ALL_PAIRS = RateTruth("all-pairs", (23.59755611, 15.16160828, 40.0))
