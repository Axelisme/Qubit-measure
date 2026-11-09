"""Rate-model coordinates and the chain rule into the 65-feature engine.

q holds three pair rates, six D0 simplex coordinates, then the same 14 shared
coordinates as the curve fit. D0 rows use a permuted simplex chart. Its largest
entry occupies slot 0, so even a pure pi-swap row has a regular chart. Only
slot 2's vertex is singular. Stages reselect the slots without changing D0.
"""

from dataclasses import dataclass, replace

import numpy as np

from ._cycle_coordinates import NUISANCE_NAMES, nuisance_bounds
from ._cycle_emission import simplex, simplex_coordinates, simplex_gradient
from ._cycle_engine import mean_jacobian
from ._cycle_layout import (
    ROW_SIZE,
    ROW_THERMAL,
    SHARED,
    TABLE_DRIVE,
    TABLE_SHARED,
    shifted,
)
from ._rate_equation import Spectrum, propagator, propagator_jacobian, spectrum
from .model import Array

RATE_NAMES = ("rate_01", "rate_02", "rate_12")
D0 = slice(3, 9)
SHARED_RATE = slice(D0.stop, D0.stop + ROW_SIZE)
THERMAL_RATE = shifted(ROW_THERMAL, SHARED_RATE.start)
SIZE = SHARED_RATE.stop
PI_FLOOR = 1e-6
RATE_MAX = 1e3
Slots = tuple[tuple[int, ...], ...]


def dominant_slots(d0: Array) -> Slots:
    """Put each row's largest probability first, other states in index order."""
    return tuple(
        (int(np.argmax(row)), *(k for k in range(3) if k != np.argmax(row)))
        for row in d0
    )


@dataclass(frozen=True)
class RateCoordinates:
    """Axis and row simplex slot permutations for one rate parameter vector."""

    axis: Array
    slots: Slots

    @property
    def names(self) -> tuple[str, ...]:
        return (
            RATE_NAMES
            + tuple(
                f"drive0_{i}_{''.join(map(str, slot))}_{c}"
                for i, slot in enumerate(self.slots)
                for c in ("s", "f")
            )
            + NUISANCE_NAMES[SHARED]
        )

    def d0(self, q: Array) -> Array:
        result = np.empty((3, 3))
        for row, slot in enumerate(self.slots):
            result[row, list(slot)] = simplex(*q[3 + 2 * row : 5 + 2 * row])
        return result

    def d0_gradient(self, q: Array) -> Array:
        result = np.empty((3, 3, 2))
        for row, slot in enumerate(self.slots):
            result[row, list(slot)] = simplex_gradient(*q[3 + 2 * row : 5 + 2 * row])
        return result

    def thermal(self, q: Array) -> Array:
        return simplex(*q[THERMAL_RATE])

    def spectrum(self, q: Array) -> Spectrum:
        return spectrum(q[:3], self.thermal(q))

    def table(self, q: Array) -> tuple[Array, Array]:
        relaxation = propagator(self.spectrum(q), self.axis)
        return self.d0(q) @ relaxation, np.tile(q[SHARED_RATE], (len(self.axis), 1))

    def mean_jacobian(self, q: Array) -> tuple[Array, Array]:
        relax, d_rate, d_pi = propagator_jacobian(self.spectrum(q), self.axis)
        d0 = self.d0(q)
        rows = np.tile(q[SHARED_RATE], (len(self.axis), 1))
        means, local = mean_jacobian(d0 @ relax, rows)

        def through_drive(change: Array) -> Array:
            pairs = change[:, :, :2].reshape(len(self.axis), 6, -1)
            return local[:, :, TABLE_DRIVE] @ pairs

        jac = np.zeros((len(self.axis), 65, SIZE))
        jac[:, :, :3] = through_drive(np.einsum("ij,njkp->nikp", d0, d_rate))
        gradient = self.d0_gradient(q)
        for row in range(3):
            change = gradient[row].T @ relax[:, :, :2]
            block = local[:, :, 2 * row : 2 * row + 2]
            jac[:, :, 3 + 2 * row : 5 + 2 * row] = block @ np.swapaxes(change, 1, 2)
        jac[:, :, SHARED_RATE] = local[:, :, TABLE_SHARED]
        # Thermal changes both the cycle start (engine term) and the generator.
        jac[:, :, THERMAL_RATE] += through_drive(
            np.einsum("ij,njkm->nikm", d0, d_pi @ simplex_gradient(*q[THERMAL_RATE]))
        )
        return means, jac

    def bounds(self) -> tuple[Array, Array]:
        shared_lo, shared_hi = nuisance_bounds()
        lo = np.r_[np.zeros(9), shared_lo[SHARED]]
        hi = np.r_[np.full(3, RATE_MAX), np.ones(6), shared_hi[SHARED]]
        # The spectral similarity divides by sqrt(pi); stay inside the simplex.
        lo[THERMAL_RATE], hi[THERMAL_RATE] = PI_FLOOR, 1 - PI_FLOOR
        return lo, hi

    def reslot(self, q: Array) -> tuple["RateCoordinates", Array]:
        d0 = self.d0(q)
        moved = replace(self, slots=dominant_slots(d0))
        result = q.copy()
        result[D0] = moved.pack_d0(d0)
        return moved, result

    def pack_d0(self, d0: Array) -> Array:
        return np.concatenate(
            [
                np.clip(simplex_coordinates(d0[row, list(slot)]), 0.0, 1.0)
                for row, slot in enumerate(self.slots)
            ]
        )

    def eigen_time_gradient(self, q: Array, mode: int) -> Array:
        """Hellmann–Feynman followed by d(-1/lambda) = d(lambda)/lambda²."""
        spec = self.spectrum(q)
        vector = spec.vectors[:, mode]
        gradient = np.zeros(SIZE)
        gradient[:3] = np.einsum("i,pij,j->p", vector, spec.rate_directions, vector)
        d_pi = np.einsum("i,mij,j->m", vector, spec.thermal_directions, vector)
        gradient[THERMAL_RATE] = d_pi @ simplex_gradient(*q[THERMAL_RATE])
        return gradient / spec.values[mode] ** 2
