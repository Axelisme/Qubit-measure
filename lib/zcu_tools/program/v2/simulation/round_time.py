"""Physical duration of a simulation request round (ADR-0084)."""

from zcu_tools.simulate.boundary import AcquireRequest


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
