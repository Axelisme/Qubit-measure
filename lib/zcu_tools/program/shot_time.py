"""Shot duration arithmetic shared by estimates and simulation pacing."""

import math

import numpy as np
from numpy.typing import ArrayLike

from zcu_tools.core.simulation import AcquireRequest


def request_round_time_s(request: AcquireRequest) -> float:
    """Return physical seconds for one round of an integer-time request.

    Sum all template point durations and one relax delay per point, then
    multiply by request.reps. Use Python integers so totals may exceed int64
    and the per-request ns bound. TimeBase performs the sole conversion to
    seconds. Invalid nonpositive reps raise ValueError; request construction
    owns per-point time validation. No overhead or wall time is included.
    """
    if type(request.reps) is not int or request.reps <= 0:
        raise ValueError("reps must be a positive built-in integer")
    ticks = sum(
        sum(int(t) for t in template.duration_ticks)
        + len(template.points) * int(request.relax_delay_ticks)
        for template in request.templates
    )
    return request.time_base.seconds(request.reps * ticks)


def shot_time_s(
    *, rounds: int, reps: int, shot_lengths_us: ArrayLike, relax_delay_us: float
) -> float:
    """Return shot seconds for a full acquisition, excluding fixed overhead.

    rounds and reps are positive built-in integers (not bool). shot_lengths_us
    is a nonempty one-dimensional array-like of finite nonnegative durations in
    microseconds, one per sweep point, excluding relaxation. relax_delay_us is
    finite nonnegative relaxation in microseconds, applied after every shot.
    Inputs are not modified. The float result is rounds * reps * the sum of
    each point duration plus relaxation, converted from microseconds to seconds.
    Acquisition overhead and flux ramp/settle time are not included.

    Raises ValueError for invalid counts, point-array shape, negative or
    nonfinite durations. NumPy conversion errors propagate when point durations
    cannot be converted to float64.
    """
    for name, count in (("rounds", rounds), ("reps", reps)):
        if type(count) is not int or count <= 0:
            raise ValueError(f"{name} must be a positive built-in integer")
    if not math.isfinite(relax_delay_us) or relax_delay_us < 0:
        raise ValueError("relax_delay_us must be finite and nonnegative")
    lengths = np.asarray(shot_lengths_us, dtype=np.float64)
    if (
        lengths.ndim != 1
        or lengths.size == 0
        or not np.all(np.isfinite(lengths))
        or np.any(lengths < 0)
    ):
        raise ValueError(
            "shot_lengths_us must be a nonempty 1D array of finite nonnegative durations"
        )
    return float(
        rounds * reps * (np.sum(lengths) + lengths.size * relax_delay_us) * 1e-6
    )
