"""Choose the T1-based recovery time between overnight Echo acquisitions."""

from collections.abc import Iterable
from typing import Literal


def echo_relaxation(
    accepted_t1_us: Iterable[float | None],
    *,
    t1_guess_us: float,
    relax_factor: float,
    relax_min_us: float,
) -> tuple[float, Literal["accepted_t1", "guess"]]:
    """Return Echo relaxation in us and the T1 estimate source.

    Supply this round's accepted participant lifetimes in us; rejected or
    failed fits are None. Use the largest accepted value, or t1_guess_us when
    none is available. Multiply by relax_factor and floor at relax_min_us.
    Inputs come from validated tunables and accepted fits; no history is used.
    """
    t1_us = max((value for value in accepted_t1_us if value is not None), default=None)
    source: Literal["accepted_t1", "guess"] = "accepted_t1"
    if t1_us is None:
        t1_us = t1_guess_us
        source = "guess"
    return max(relax_min_us, relax_factor * t1_us), source
