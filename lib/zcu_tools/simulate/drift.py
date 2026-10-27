"""Exact log-parameter Ornstein-Uhlenbeck drift with caller-owned RNG."""

import math

import numpy as np


class LogOU:
    """A positive parameter whose logarithm follows an exact OU transition.

    mean is the positive reference value: the stationary log mean is log(mean),
    not the linear arithmetic mean. sigma is stationary log standard deviation
    (>=0); tau_s is positive correlation time in seconds. rng is exclusively
    owned by this process. The first sample fixes its time origin and returns
    mean, so startup does not draw a stationary random initial condition.
    """

    def __init__(
        self, mean: float, *, sigma: float, tau_s: float, rng: np.random.Generator
    ) -> None:
        if not math.isfinite(mean) or mean <= 0:
            raise ValueError("mean must be finite and positive")
        if not math.isfinite(sigma) or sigma < 0:
            raise ValueError("sigma must be finite and nonnegative")
        if not math.isfinite(tau_s) or tau_s <= 0:
            raise ValueError("tau_s must be finite and positive")
        self._mean = mean
        self._sigma = sigma
        self._tau_s = tau_s
        self._rng = rng
        self._deviation = 0.0
        self._time: float | None = None

    def sample(self, time_s: float) -> float:
        """Return the loss value at finite nondecreasing simulated seconds.

        Equal times return unchanged values without consuming RNG. Backward or
        nonfinite time raises ValueError. Arbitrarily large gaps use the exact
        stationary limit; tiny gaps retain precision through expm1.
        """
        if not math.isfinite(time_s) or (
            self._time is not None and time_s < self._time
        ):
            raise ValueError("time must be finite and nondecreasing")
        if self._time is not None and time_s > self._time and self._sigma > 0:
            gap = (time_s - self._time) / self._tau_s
            self._deviation = (
                math.exp(-gap) * self._deviation
                + self._sigma
                * math.sqrt(-math.expm1(-2 * gap))
                * self._rng.standard_normal()
            )
        self._time = time_s
        return self._mean * math.exp(self._deviation)
