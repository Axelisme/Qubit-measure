"""Run-lived scalar feedback capabilities.

Feedback is keyed by placed-node identity and semantic slot key. The generic
layer provides scalar estimators/controllers only; use sites decide whether an
estimate/proposal is acceptable and how to apply clamps or safety gates.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from zcu_tools.utils.math import IDWInterpolation


@runtime_checkable
class ScalarEstimator(Protocol):
    """A scalar flux-indexed estimator."""

    def observe(self, flux: float, value: float) -> None: ...
    def replace_observations(
        self, observations: Iterable[tuple[float, float]]
    ) -> None: ...
    def estimate(self, flux: float) -> FeedbackSample | None: ...


@runtime_checkable
class ScalarController(Protocol):
    """A scalar actuator controller."""

    def latest(self) -> FeedbackSample | None: ...
    def propose(self, current: float, normalized_error: float) -> FeedbackSample: ...


@dataclass(frozen=True)
class FeedbackSample:
    """A finite value, confidence in [0, 1], and nonnegative age in queries.

    Invalid values raise RuntimeError. Confidence decays as exp(-age/decay_points).
    """

    value: float
    confidence: float
    age_queries: int

    def __post_init__(self) -> None:
        value = _finite("sample value", self.value)
        confidence = _finite("confidence", self.confidence)
        if not 0.0 <= confidence <= 1.0:
            raise RuntimeError("feedback confidence must be between 0 and 1")
        if type(self.age_queries) is not int:
            raise RuntimeError("feedback age_queries must be a non-negative integer")
        if self.age_queries < 0:
            raise RuntimeError("feedback age_queries must be a non-negative integer")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "confidence", confidence)


def _finite(name: str, value: float) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise RuntimeError(f"feedback {name} must be finite")
    return out


def _positive_finite(name: str, value: float) -> float:
    out = _finite(name, value)
    if out <= 0.0:
        raise RuntimeError(f"feedback {name} must be positive")
    return out


def _positive_int(name: str, value: float) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise RuntimeError(f"feedback {name} must be a positive integer")
    if value <= 0:
        raise RuntimeError(f"feedback {name} must be a positive integer")
    return value


def _confidence(age_queries: int, decay_points: float) -> float:
    if age_queries < 0:
        raise RuntimeError("feedback age_queries must be a non-negative integer")
    decay = _positive_finite("decay_points", decay_points)
    return math.exp(-float(age_queries) / decay)


def _sample(value: float, age_queries: int, decay_points: float) -> FeedbackSample:
    return FeedbackSample(
        value=_finite("sample value", value),
        confidence=_confidence(age_queries, decay_points),
        age_queries=age_queries,
    )


@dataclass
class LastGoodEstimator:
    """Keep the latest finite observation, decaying confidence per query.

    decay_points is the positive query-age e-folding time (default 3).
    observe resets age; replace_observations replaces history and resets age.
    estimate returns None before observation; flux is unused.
    """

    decay_points: float = 3.0
    _value: float | None = None
    _query_count: int = 0
    _last_update_query: int = 0

    def __post_init__(self) -> None:
        self.decay_points = _positive_finite("decay_points", self.decay_points)

    def observe(self, flux: float, value: float) -> None:
        del flux
        self._value = _finite("observation", value)
        self._last_update_query = self._query_count

    def replace_observations(self, observations: Iterable[tuple[float, float]]) -> None:
        latest: float | None = None
        for flux, value in observations:
            _finite("flux", flux)
            latest = _finite("observation", value)
        self._value = latest
        self._query_count = 0
        self._last_update_query = 0

    def estimate(self, flux: float) -> FeedbackSample | None:
        del flux
        if self._value is None:
            return None
        age_queries = max(0, self._query_count - self._last_update_query)
        sample = _sample(self._value, age_queries, self.decay_points)
        self._query_count += 1
        return sample


@dataclass
class IdwEstimator:
    """Interpolate finite (flux, value) observations with inverse distance weights.

    k is the positive neighbor count; epsilon is the positive distance floor.
    decay_points is the positive query-age e-folding time (default 3).
    Empty history estimates return None; invalid inputs raise RuntimeError.
    """

    k: int = 4
    epsilon: float = 1e-6
    decay_points: float = 3.0
    _idw: IDWInterpolation = field(init=False)
    _observations: int = 0
    _query_count: int = 0
    _last_update_query: int = 0

    def __post_init__(self) -> None:
        self.k = _positive_int("idw_k", self.k)
        self.epsilon = _positive_finite("idw_epsilon", self.epsilon)
        self.decay_points = _positive_finite("decay_points", self.decay_points)
        self._idw = IDWInterpolation(k=self.k, epsilon=self.epsilon)

    def observe(self, flux: float, value: float) -> None:
        self._idw.update(_finite("flux", flux), _finite("observation", value))
        self._observations += 1
        self._last_update_query = self._query_count

    def replace_observations(self, observations: Iterable[tuple[float, float]]) -> None:
        new_idw = IDWInterpolation(k=self.k, epsilon=self.epsilon)
        count = 0
        for flux, value in observations:
            new_idw.update(_finite("flux", flux), _finite("observation", value))
            count += 1
        self._idw = new_idw
        self._observations = count
        self._query_count = 0
        self._last_update_query = 0

    def estimate(self, flux: float) -> FeedbackSample | None:
        if self._observations == 0:
            return None
        age_queries = max(0, self._query_count - self._last_update_query)
        sample = _sample(
            float(self._idw.predict(_finite("flux", flux))),
            age_queries,
            self.decay_points,
        )
        self._query_count += 1
        return sample


@dataclass
class LogStepController:
    """Propose current * exp(step_gain * normalized_error), without clamps.

    step_gain and decay_points are positive finite control gain and query-age
    e-folding time. latest returns None before propose, then increments age on
    each query. propose resets age. Invalid/overflowed values raise RuntimeError.
    """

    step_gain: float = 1.0
    decay_points: float = 3.0
    _latest: float | None = None
    _query_count: int = 0
    _last_update_query: int = 0

    def __post_init__(self) -> None:
        self.step_gain = _positive_finite("step_gain", self.step_gain)
        self.decay_points = _positive_finite("decay_points", self.decay_points)

    def latest(self) -> FeedbackSample | None:
        if self._latest is None:
            return None
        age_queries = max(0, self._query_count - self._last_update_query)
        sample = _sample(self._latest, age_queries, self.decay_points)
        self._query_count += 1
        return sample

    def propose(self, current: float, normalized_error: float) -> FeedbackSample:
        current_value = _positive_finite("current actuator", current)
        error = _finite("normalized_error", normalized_error)
        try:
            proposal = current_value * math.exp(self.step_gain * error)
        except OverflowError as exc:
            raise RuntimeError(
                "feedback controller proposal must be positive and finite"
            ) from exc
        if not np.isfinite(proposal) or proposal <= 0.0:
            raise RuntimeError(
                "feedback controller proposal must be positive and finite"
            )
        self._latest = float(proposal)
        self._last_update_query = self._query_count
        return _sample(self._latest, 0, self.decay_points)
