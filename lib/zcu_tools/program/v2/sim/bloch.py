"""Two-level-system (TLS) optical Bloch-equation propagator.

Self-contained physics core for the MockSoc SimEngine: it turns each
piecewise-constant control segment into an affine map on the Bloch vector via a
4x4 augmented-matrix matrix exponential, and chains segments into a single
evolution. It depends only on numpy/scipy and imports nothing from the rest of
the project (deliberately leaf, so params/lowering/readout/engine can build on
it without a cycle).

Conventions (fixed by the analytic-limit tests in
``tests/program/v2/sim/test_bloch_limits.py``):
  - Bloch vector ``v = (x, y, z)`` with ``z = <sigma_z>``; ground ``|g>`` sits at
    ``z = -1`` and excited ``|e>`` at ``z = +1``. Excited population is therefore
    ``P_e = (1 + z) / 2``.
  - Rotating frame + RWA. ``delta = omega_qubit - omega_drive`` (qubit minus
    drive). ``omega`` is the Rabi rate, ``phase`` the drive phase.
  - Relaxation enters as rates ``gamma1 = 1/T1`` and ``gamma2 = 1/T2``; passing
    ``T1``/``T2`` as ``None`` (or non-positive) means the infinite-time limit
    ``gamma = 0`` (no decay), which avoids any division by zero.

Affine ODE ``v' = M v + b`` with::

    M = [[-gamma2,   -delta,      omega*sin(phase)],
         [  delta,   -gamma2,    -omega*cos(phase)],
         [-omega*sin(phase), omega*cos(phase), -gamma1]]
    b = [0, 0, z_eq * gamma1],   z_eq = 2*thermal_pop - 1

The segment propagator is ``P = expm(G * t)`` where ``G`` is the 4x4 augmented
generator ``[[M, b], [0, 0]]``; then ``[v_out, 1] = P @ [v_in, 1]``.
"""

from __future__ import annotations

from typing import NamedTuple
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm


def _rate(time_const: float | None) -> float:
    """Convert a coherence time into a decay rate, treating None/<=0 as 0.

    Fast-fail intent: a finite positive time gives ``1/time``; the infinite-time
    limit (no decay) is requested explicitly via ``None`` or a non-positive value
    and yields rate 0, so the generator never divides by zero.
    """

    if time_const is None or time_const <= 0.0:
        return 0.0
    return 1.0 / float(time_const)


def bloch_generator(
    omega: float,
    delta: float,
    phase: float,
    t1: float | None,
    t2: float | None,
    thermal_pop: float = 0.0,
) -> NDArray[np.float64]:
    """Return the 4x4 augmented generator ``G = [[M, b], [0, 0]]``.

    ``omega`` is the Rabi rate, ``delta = omega_qubit - omega_drive`` the
    detuning, ``phase`` the drive phase. ``t1``/``t2`` are coherence times (None
    or <=0 means no decay). ``thermal_pop`` is the equilibrium excited
    population, so ``z_eq = 2*thermal_pop - 1``.
    """

    gamma1 = _rate(t1)
    gamma2 = _rate(t2)
    sin_p = np.sin(phase)
    cos_p = np.cos(phase)
    z_eq = 2.0 * thermal_pop - 1.0

    gen = np.zeros((4, 4), dtype=np.float64)
    gen[0, 0] = -gamma2
    gen[0, 1] = -delta
    gen[0, 2] = omega * sin_p
    gen[1, 0] = delta
    gen[1, 1] = -gamma2
    gen[1, 2] = -omega * cos_p
    gen[2, 0] = -omega * sin_p
    gen[2, 1] = omega * cos_p
    gen[2, 2] = -gamma1
    gen[2, 3] = z_eq * gamma1  # b vector (affine drive toward thermal equilibrium)
    return gen


def segment_propagator(
    omega: float,
    delta: float,
    phase: float,
    t: float,
    t1: float | None,
    t2: float | None,
    thermal_pop: float = 0.0,
) -> NDArray[np.float64]:
    """Return the 4x4 affine propagator ``expm(G * t)`` for one segment.

    Applying it as ``[v_out, 1] = P @ [v_in, 1]`` advances the Bloch vector by
    duration ``t`` under constant ``(omega, delta, phase)`` and the given
    relaxation.
    """

    gen = bloch_generator(omega, delta, phase, t1, t2, thermal_pop)
    # expm's stub returns a loose float union; coerce to the float64 we promise.
    return np.asarray(expm(gen * float(t)), dtype=np.float64)


class Segment(NamedTuple):
    """One piecewise-constant control segment for :func:`evolve`."""

    omega: float
    delta: float
    phase: float
    t: float
    t1: float | None
    t2: float | None
    thermal_pop: float = 0.0


def evolve(
    v0: NDArray[np.float64],
    segments: Sequence[Segment],
) -> NDArray[np.float64]:
    """Evolve an initial Bloch vector through a sequence of segments.

    Segments are applied in list order: the first segment acts first, so the
    combined 4x4 propagator is the reverse-order product (later segment on the
    left). Returns the final 3-vector.
    """

    v0 = np.asarray(v0, dtype=np.float64)
    if v0.shape != (3,):
        raise ValueError(f"v0 must have shape (3,), got {v0.shape}")

    aug = np.empty(4, dtype=np.float64)
    aug[:3] = v0
    aug[3] = 1.0

    for seg in segments:
        prop = segment_propagator(
            seg.omega, seg.delta, seg.phase, seg.t, seg.t1, seg.t2, seg.thermal_pop
        )
        aug = prop @ aug

    return aug[:3].copy()


def ground_state(thermal_pop: float = 0.0) -> NDArray[np.float64]:
    """Return the thermal steady state ``(0, 0, z_eq)`` with ``z_eq = 2*p - 1``.

    With ``thermal_pop = 0`` this is the pure ground state ``z = -1``.
    """

    return np.array([0.0, 0.0, 2.0 * thermal_pop - 1.0], dtype=np.float64)


def excited_population(v: NDArray[np.float64]) -> float:
    """Return excited-state population ``P_e = (1 + z) / 2`` from a Bloch vector."""

    return float((1.0 + v[2]) / 2.0)
