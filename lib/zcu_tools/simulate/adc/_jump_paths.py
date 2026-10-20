"""Receiver-independent continuous-time Markov paths for one shot group."""

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from .model import RealArray


class PathStep(NamedTuple):
    """One wave of path segments, including the final segment of each shot.

    shots: int64 active shot indices. right: emitted-ns segment ends, with the
    terminal boundary advanced one ULP to include the final sample. targets:
    int64 destination states, or -1 for terminal segments. Arrays are owned by
    this path, shared read-only by all cavity/receiver workspaces.
    """

    shots: NDArray[np.int64]
    right: RealArray
    targets: NDArray[np.int64]


class JumpPaths:
    """Draw complete CTMC paths once for all carriers, receivers and windows.

    initial: valid int64 (shot,) model-state indices, retained independently.
    start/end: finite emitted-ns population origin and latest sample boundary.
    rates: validated [target,source] directional rates. rng: caller-local stream.
    steps: tuple of PathStep waves with no jump-count conditioning or cutoff.
    No field, receiver noise or interpolation is stored. Invalid rate totals or
    unresolved floating-point jump times raise ValueError.
    """

    def __init__(
        self,
        initial: NDArray[np.int64],
        start: float,
        end: float,
        rates: RealArray,
        rng: np.random.Generator,
    ) -> None:
        self.initial = initial.copy()
        self.start = start
        states = initial.copy()
        left = np.full(len(states), start, dtype=float)
        active = np.arange(len(states), dtype=np.int64)
        totals = rates.sum(axis=0)
        if not np.isfinite(totals).all():
            raise ValueError("transition rate totals must be finite")
        steps = []
        while len(active):
            total = totals[states[active]]
            waiting = np.full(len(active), np.inf)
            moving = total > 0
            waiting[moving] = rng.exponential(1 / total[moving])
            jumps = left[active] + waiting
            if np.any(moving & (jumps <= left[active])):
                raise ValueError(
                    "transition times are below floating-point time resolution"
                )
            continuing = jumps < end
            right = np.where(continuing, jumps, np.nextafter(end, np.inf))
            targets = np.full(len(active), -1, dtype=np.int64)
            current = states[active[continuing]]
            cumulative = np.cumsum(rates[:, current], axis=0)
            uniforms = rng.random(len(current)) * totals[current]
            targets[continuing] = (uniforms[None, :] >= cumulative[:-1]).sum(axis=0)
            steps.append(PathStep(active, right, targets))
            active = active[continuing]
            states[active] = targets[continuing]
            left[active] = jumps[continuing]
        self.steps = tuple(steps)
