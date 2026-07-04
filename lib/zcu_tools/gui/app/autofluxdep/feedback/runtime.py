"""Run-lived scalar feedback capabilities.

Feedback is keyed by placed-node identity and semantic slot key. The generic
layer provides scalar estimators/controllers only; use sites decide whether an
estimate/proposal is acceptable and how to apply clamps or safety gates.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, cast, runtime_checkable

import numpy as np

from zcu_tools.gui.app.autofluxdep.cfg import FloatSpec, IntSpec, ScalarSpec
from zcu_tools.gui.app.autofluxdep.cfg.schema import str_choice_spec
from zcu_tools.gui.app.autofluxdep.nodes.defaults import (
    GenerationField,
    generation_field,
)
from zcu_tools.utils.math import IDWInterpolation

FeedbackKind = Literal["estimator", "controller"]
EstimatorStrategy = Literal["idw", "last_good"]
ControllerStrategy = Literal["log_step"]


@runtime_checkable
class ScalarEstimator(Protocol):
    """A scalar flux-indexed estimator."""

    def observe(self, flux: float, value: float) -> None: ...
    def estimate(self, flux: float) -> float | None: ...


@runtime_checkable
class ScalarController(Protocol):
    """A scalar actuator controller."""

    def latest(self) -> float | None: ...
    def propose(self, current: float, normalized_error: float) -> float: ...


@dataclass(frozen=True)
class FeedbackSlotDecl:
    """Node-owned declaration for one semantic feedback slot."""

    key: str
    kind: FeedbackKind
    prefix: str
    default_enabled: bool = True
    default_strategy: str | None = None
    default_idw_k: int = 4
    default_idw_epsilon: float = 1e-6
    default_step_gain: float = 1.0

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("feedback slot key must be non-empty")
        if not self.prefix:
            raise ValueError("feedback slot prefix must be non-empty")
        default_strategy = self.default_strategy
        if self.kind == "estimator":
            if default_strategy is None:
                object.__setattr__(self, "default_strategy", "idw")
            elif default_strategy not in ("idw", "last_good"):
                raise ValueError(
                    f"unsupported estimator strategy: {default_strategy!r}"
                )
        elif self.kind == "controller":
            if default_strategy is None:
                object.__setattr__(self, "default_strategy", "log_step")
            elif default_strategy != "log_step":
                raise ValueError(
                    f"unsupported controller strategy: {default_strategy!r}"
                )
        else:
            raise ValueError(f"unsupported feedback slot kind: {self.kind!r}")

    def field_name(self, suffix: str) -> str:
        return f"{self.prefix}_{suffix}"


def feedback_generation_fields(slot: FeedbackSlotDecl) -> tuple[GenerationField, ...]:
    """Return generation fields for ``slot`` under ``generation.feedback``."""

    fields: list[GenerationField] = [
        generation_field(
            slot.field_name("enabled"),
            slot.field_name("enabled"),
            ScalarSpec(label=slot.field_name("enabled"), type=bool),
            slot.default_enabled,
            group="feedback",
        )
    ]
    if slot.kind == "estimator":
        fields.extend(
            (
                generation_field(
                    slot.field_name("strategy"),
                    slot.field_name("strategy"),
                    str_choice_spec(slot.field_name("strategy"), ("idw", "last_good")),
                    str(slot.default_strategy),
                    group="feedback",
                ),
                generation_field(
                    slot.field_name("idw_k"),
                    slot.field_name("idw_k"),
                    IntSpec(label=slot.field_name("idw_k")),
                    slot.default_idw_k,
                    group="feedback",
                ),
                generation_field(
                    slot.field_name("idw_epsilon"),
                    slot.field_name("idw_epsilon"),
                    FloatSpec(label=slot.field_name("idw_epsilon")),
                    slot.default_idw_epsilon,
                    group="feedback",
                ),
            )
        )
    else:
        fields.extend(
            (
                generation_field(
                    slot.field_name("strategy"),
                    slot.field_name("strategy"),
                    str_choice_spec(slot.field_name("strategy"), ("log_step",)),
                    str(slot.default_strategy),
                    group="feedback",
                ),
                generation_field(
                    slot.field_name("step_gain"),
                    slot.field_name("step_gain"),
                    FloatSpec(label=slot.field_name("step_gain")),
                    slot.default_step_gain,
                    group="feedback",
                ),
            )
        )
    return tuple(fields)


def _finite(name: str, value: Any) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise RuntimeError(f"feedback {name} must be finite")
    return out


def _positive_finite(name: str, value: Any) -> float:
    out = _finite(name, value)
    if out <= 0.0:
        raise RuntimeError(f"feedback {name} must be positive")
    return out


def _positive_int(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise RuntimeError(f"feedback {name} must be a positive integer")
    if value <= 0:
        raise RuntimeError(f"feedback {name} must be a positive integer")
    return value


@dataclass
class LastGoodEstimator:
    _value: float | None = None

    def observe(self, flux: float, value: float) -> None:
        del flux
        self._value = _finite("observation", value)

    def estimate(self, flux: float) -> float | None:
        del flux
        return self._value


@dataclass
class IdwEstimator:
    k: int = 4
    epsilon: float = 1e-6
    _idw: IDWInterpolation = field(init=False)
    _observations: int = 0

    def __post_init__(self) -> None:
        self.k = _positive_int("idw_k", self.k)
        self.epsilon = _positive_finite("idw_epsilon", self.epsilon)
        self._idw = IDWInterpolation(k=self.k, epsilon=self.epsilon)

    def observe(self, flux: float, value: float) -> None:
        self._idw.update(_finite("flux", flux), _finite("observation", value))
        self._observations += 1

    def estimate(self, flux: float) -> float | None:
        if self._observations == 0:
            return None
        return float(self._idw.predict(_finite("flux", flux)))


@dataclass
class LogStepController:
    """Stateless log-domain scalar actuator controller with run-lived latest value."""

    step_gain: float = 1.0
    _latest: float | None = None

    def __post_init__(self) -> None:
        self.step_gain = _positive_finite("step_gain", self.step_gain)

    def latest(self) -> float | None:
        return self._latest

    def propose(self, current: float, normalized_error: float) -> float:
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
        return self._latest


FeedbackCapability = ScalarEstimator | ScalarController


@dataclass
class FeedbackView:
    """Placement-scoped feedback slot map exposed to a Node."""

    _slots: Mapping[str, FeedbackCapability | None] = field(default_factory=dict)

    def estimator(self, key: str) -> ScalarEstimator | None:
        slot = self._require(key)
        if slot is None:
            return None
        if not isinstance(slot, ScalarEstimator):
            raise TypeError(f"feedback slot {key!r} is not an estimator")
        return slot

    def controller(self, key: str) -> ScalarController | None:
        slot = self._require(key)
        if slot is None:
            return None
        if not isinstance(slot, ScalarController):
            raise TypeError(f"feedback slot {key!r} is not a controller")
        return slot

    def _require(self, key: str) -> FeedbackCapability | None:
        try:
            return self._slots[key]
        except KeyError as exc:
            raise KeyError(f"undeclared feedback slot {key!r}") from exc


@dataclass
class FeedbackRuntime:
    """Run-lived feedback capability registry."""

    _placements: Mapping[str, Mapping[str, FeedbackCapability | None]] = field(
        default_factory=dict
    )

    def view_for(self, placement_name: str) -> FeedbackView:
        return FeedbackView(self._placements.get(placement_name, {}))


def build_feedback_runtime(providers: Iterable[Any], md: Any = None) -> FeedbackRuntime:
    placements: dict[str, dict[str, FeedbackCapability | None]] = {}
    for provider in providers:
        slots = cast(
            tuple[FeedbackSlotDecl, ...],
            tuple(getattr(provider.builder, "feedback_slots", ())),
        )
        if not slots:
            continue
        knobs = provider.schema.lower(None, md=md)
        placement_slots: dict[str, FeedbackCapability | None] = {}
        for slot in slots:
            placement_slots[slot.key] = _build_capability(slot, knobs)
        placements[provider.name] = placement_slots
    return FeedbackRuntime(placements)


def _build_capability(
    slot: FeedbackSlotDecl, knobs: Mapping[str, Any]
) -> FeedbackCapability | None:
    enabled = bool(knobs[slot.field_name("enabled")])
    if not enabled:
        return None
    strategy = str(knobs[slot.field_name("strategy")])
    if slot.kind == "estimator":
        if strategy == "last_good":
            return LastGoodEstimator()
        if strategy == "idw":
            return IdwEstimator(
                k=_positive_int(
                    slot.field_name("idw_k"), knobs[slot.field_name("idw_k")]
                ),
                epsilon=_positive_finite(
                    slot.field_name("idw_epsilon"),
                    knobs[slot.field_name("idw_epsilon")],
                ),
            )
        raise RuntimeError(f"unsupported estimator feedback strategy: {strategy!r}")
    if strategy == "log_step":
        return LogStepController(
            step_gain=_positive_finite(
                slot.field_name("step_gain"), knobs[slot.field_name("step_gain")]
            )
        )
    raise RuntimeError(f"unsupported controller feedback strategy: {strategy!r}")
