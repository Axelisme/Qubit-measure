"""Independent lifetime-history projections for coherence workflows."""

from typing import Literal

type LifetimeKind = Literal["t1", "t2ramsey", "t2echo"]
type LifetimeHistory = list[tuple[float, float, float]]


def lifetime_guess(initial_us: float, history: LifetimeHistory) -> float:
    """Return the alpha=.5 EWMA of accepted (flux, lifetime_us, error_us).

    initial_us is this lifetime's validated positive seed. Only that lifetime's
    accepted history enters its grid; neither the input list nor seed is changed.
    """
    guess = initial_us
    for _, lifetime, _error in history:
        guess = 0.5 * lifetime + 0.5 * guess
    return guess
