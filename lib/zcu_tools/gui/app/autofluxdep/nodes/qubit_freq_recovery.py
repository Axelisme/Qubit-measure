"""Fail-triggered physical recovery for the qubit_freq node."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np

from zcu_tools.gui.app.autofluxdep.feedback import ScalarEstimator
from zcu_tools.gui.app.autofluxdep.nodes.builder import RunEnv
from zcu_tools.gui.app.autofluxdep.tools import Predictor, Tools
from zcu_tools.simulate.fluxonium.physical_fit import (
    FluxoniumLocalFitResult,
    fit_local_fluxonium_model,
)

logger = logging.getLogger(__name__)

PHYSICAL_RECOVERY_MODE_OFF = "off"
PHYSICAL_RECOVERY_MODE_FAIL_TRIGGERED_FIT = "fail_triggered_fit"
PhysicalRecoveryMode = Literal["off", "fail_triggered_fit"]

DEFAULT_PHYSICAL_RECOVERY_MIN_POINTS = 10
DEFAULT_PHYSICAL_RECOVERY_MAX_POINTS = 30
DEFAULT_PHYSICAL_RECOVERY_MAX_CENTER_SHIFT_MHZ = 150.0
DEFAULT_PHYSICAL_RECOVERY_MAX_RMS_MHZ = 50.0


@dataclass(frozen=True)
class TrustedFrequencyPoint:
    flux: float
    frequency_mhz: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "flux", _finite("trusted flux", self.flux))
        object.__setattr__(
            self,
            "frequency_mhz",
            _finite("trusted frequency", self.frequency_mhz),
        )


@dataclass(frozen=True)
class PhysicalRecoveryConfig:
    mode: PhysicalRecoveryMode
    min_points: int
    max_points: int
    max_center_shift_mhz: float
    max_rms_mhz: float

    @property
    def enabled(self) -> bool:
        return self.mode == PHYSICAL_RECOVERY_MODE_FAIL_TRIGGERED_FIT


@dataclass(frozen=True)
class PhysicalRecoveryAttempt:
    trigger: str
    accepted: bool
    reason: str
    n_points: int
    base_rms_mhz: float
    fitted_rms_mhz: float
    center_shift_mhz: float


@dataclass
class QubitFreqRecoveryState:
    history: list[TrustedFrequencyPoint] = field(default_factory=list)
    fail_streak: int = 0
    overlay: Predictor | None = None
    last_attempt: PhysicalRecoveryAttempt | None = None


def recovery_config_from_knobs(knobs: Mapping[str, Any]) -> PhysicalRecoveryConfig:
    mode = str(knobs["physical_recovery_mode"])
    if mode not in (
        PHYSICAL_RECOVERY_MODE_OFF,
        PHYSICAL_RECOVERY_MODE_FAIL_TRIGGERED_FIT,
    ):
        raise RuntimeError(f"unsupported qubit_freq physical_recovery_mode: {mode!r}")

    min_points = _int(
        "physical_recovery_min_points", knobs["physical_recovery_min_points"]
    )
    max_points = _int(
        "physical_recovery_max_points", knobs["physical_recovery_max_points"]
    )
    if not (
        DEFAULT_PHYSICAL_RECOVERY_MIN_POINTS
        <= min_points
        <= max_points
        <= DEFAULT_PHYSICAL_RECOVERY_MAX_POINTS
    ):
        raise RuntimeError(
            "qubit_freq physical recovery fit points must satisfy "
            "10 <= min_points <= max_points <= 30"
        )

    return PhysicalRecoveryConfig(
        mode=cast(PhysicalRecoveryMode, mode),
        min_points=min_points,
        max_points=max_points,
        max_center_shift_mhz=_positive_finite(
            "physical_recovery_max_center_shift_mhz",
            knobs["physical_recovery_max_center_shift_mhz"],
        ),
        max_rms_mhz=_positive_finite(
            "physical_recovery_max_rms_mhz",
            knobs["physical_recovery_max_rms_mhz"],
        ),
    )


def validate_recovery_bias_policy(knobs: Mapping[str, Any]) -> None:
    cfg = recovery_config_from_knobs(knobs)
    if cfg.enabled and str(knobs["bias_update_mode"]) == "hard":
        raise RuntimeError(
            "qubit_freq physical recovery is mutually exclusive with "
            'bias_update_mode="hard"'
        )


def physical_prediction_for_make_cfg(
    env: RunEnv,
    snapshot_predict_freq: float,
    knobs: Mapping[str, Any],
) -> float:
    """Return active physical prediction before generic residual correction."""
    validate_recovery_bias_policy(knobs)
    cfg = recovery_config_from_knobs(knobs)
    if not cfg.enabled or env.tools is None:
        return float(snapshot_predict_freq)
    state = env.tools.peek_recovery_state(_placement_key(env), QubitFreqRecoveryState)
    if state is None or state.overlay is None:
        return float(snapshot_predict_freq)
    return float(state.overlay.predict_freq(env.flux))


def physical_prediction_for_residual(
    env: RunEnv,
    snapshot_predict_freq: float,
    knobs: Mapping[str, Any],
) -> float:
    return physical_prediction_for_make_cfg(env, snapshot_predict_freq, knobs)


def on_fit_failed(
    env: RunEnv,
    *,
    snapshot_predict_freq: float,
    estimator_key: str,
) -> None:
    knobs = env.knobs()
    validate_recovery_bias_policy(knobs)
    cfg = recovery_config_from_knobs(knobs)
    if not cfg.enabled or env.tools is None:
        return
    state = env.tools.recovery_state(_placement_key(env), QubitFreqRecoveryState)
    state.fail_streak += 1
    if state.fail_streak == 1:
        _attempt_recovery_fit(
            env,
            cfg,
            state,
            trigger="first_fail",
            snapshot_predict_freq=snapshot_predict_freq,
            estimator_key=estimator_key,
        )


def on_fit_succeeded(
    env: RunEnv,
    measured_freq_mhz: float,
    *,
    snapshot_predict_freq: float,
    estimator_key: str,
) -> bool:
    knobs = env.knobs()
    validate_recovery_bias_policy(knobs)
    cfg = recovery_config_from_knobs(knobs)
    if not cfg.enabled or env.tools is None:
        return False
    state = env.tools.recovery_state(_placement_key(env), QubitFreqRecoveryState)
    state.history.append(TrustedFrequencyPoint(env.flux, measured_freq_mhz))
    had_fail = state.fail_streak > 0
    reseeded = False
    if had_fail:
        reseeded = _attempt_recovery_fit(
            env,
            cfg,
            state,
            trigger="first_success_after_fail",
            snapshot_predict_freq=snapshot_predict_freq,
            estimator_key=estimator_key,
        )
    state.fail_streak = 0
    return reseeded


def select_fit_points(
    history: Iterable[TrustedFrequencyPoint],
    *,
    min_points: int = DEFAULT_PHYSICAL_RECOVERY_MIN_POINTS,
    max_points: int = DEFAULT_PHYSICAL_RECOVERY_MAX_POINTS,
) -> tuple[TrustedFrequencyPoint, ...]:
    """Select a deterministic, flux-spread subset for local physical fitting."""
    if not (
        DEFAULT_PHYSICAL_RECOVERY_MIN_POINTS
        <= min_points
        <= max_points
        <= DEFAULT_PHYSICAL_RECOVERY_MAX_POINTS
    ):
        raise RuntimeError("fit point bounds must satisfy 10 <= min <= max <= 30")

    latest_by_flux: dict[float, TrustedFrequencyPoint] = {}
    for point in history:
        trusted = TrustedFrequencyPoint(point.flux, point.frequency_mhz)
        latest_by_flux[trusted.flux] = trusted

    points = tuple(sorted(latest_by_flux.values(), key=lambda point: point.flux))
    if len(points) < min_points:
        return ()
    if len(points) <= max_points:
        return points

    first, last = points[0], points[-1]
    span = last.flux - first.flux
    if span <= 0.0:
        return ()

    low = 0.0
    high = span / float(max_points - 1)
    while len(_greedy_spacing(points, high)) > max_points and high < span:
        high *= 2.0

    best = _greedy_spacing(points, high)
    for _ in range(64):
        mid = (low + high) / 2.0
        candidate = _greedy_spacing(points, mid)
        if len(candidate) > max_points:
            low = mid
        else:
            best = candidate
            high = mid

    selected = _prefer_last_endpoint(best, points[-1], high)
    if len(selected) > max_points:
        selected = _thin_evenly(selected, max_points)
    if len(selected) < min_points:
        return ()
    return tuple(selected)


def _attempt_recovery_fit(
    env: RunEnv,
    cfg: PhysicalRecoveryConfig,
    state: QubitFreqRecoveryState,
    *,
    trigger: str,
    snapshot_predict_freq: float,
    estimator_key: str,
) -> bool:
    selected = select_fit_points(
        state.history,
        min_points=cfg.min_points,
        max_points=cfg.max_points,
    )
    if not selected:
        state.last_attempt = _attempt(
            trigger,
            False,
            "not enough trusted history",
            0,
            math.nan,
            math.nan,
            math.nan,
        )
        return False

    predictor = _active_physical_predictor(env.tools, state)
    if predictor is None:
        state.last_attempt = _attempt(
            trigger,
            False,
            "no physical predictor",
            len(selected),
            math.nan,
            math.nan,
            math.nan,
        )
        return False
    if not predictor.supports_physical_recovery():
        state.last_attempt = _attempt(
            trigger,
            False,
            "predictor does not support physical recovery",
            len(selected),
            math.nan,
            math.nan,
            math.nan,
        )
        return False

    base = predictor.physical_snapshot()
    fit = fit_local_fluxonium_model(
        base,
        ((point.flux, point.frequency_mhz) for point in selected),
    )
    candidate = _candidate_overlay(predictor, fit)
    attempt = _accept_candidate(
        env,
        cfg,
        trigger,
        selected,
        predictor,
        candidate,
        fit,
        snapshot_predict_freq=snapshot_predict_freq,
        estimator=_estimator(env, estimator_key),
    )
    state.last_attempt = attempt
    if attempt.accepted and candidate is not None:
        state.overlay = candidate
        logger.info(
            "qubit_freq physical recovery accepted @flux%d via %s: "
            "n=%d rms %.3f->%.3f shift %.3f MHz",
            env.flux_idx,
            trigger,
            attempt.n_points,
            attempt.base_rms_mhz,
            attempt.fitted_rms_mhz,
            attempt.center_shift_mhz,
        )
    else:
        logger.debug(
            "qubit_freq physical recovery rejected @flux%d via %s: %s",
            env.flux_idx,
            trigger,
            attempt.reason,
        )
    return attempt.accepted and candidate is not None


def _candidate_overlay(
    base_predictor: Predictor,
    fit: FluxoniumLocalFitResult,
) -> Predictor | None:
    if not fit.accepted or fit.fitted is None:
        return None
    try:
        return base_predictor.overlay_physical(fit.fitted)
    except Exception as exc:
        raise RuntimeError("qubit_freq physical recovery overlay failed") from exc


def _accept_candidate(
    env: RunEnv,
    cfg: PhysicalRecoveryConfig,
    trigger: str,
    selected: tuple[TrustedFrequencyPoint, ...],
    base_predictor: Predictor,
    candidate: Predictor | None,
    fit: FluxoniumLocalFitResult,
    *,
    snapshot_predict_freq: float,
    estimator: ScalarEstimator | None,
) -> PhysicalRecoveryAttempt:
    if len(selected) < cfg.min_points or len(selected) > cfg.max_points:
        return _attempt(
            trigger,
            False,
            "selected point count outside configured bounds",
            len(selected),
            fit.base_rms_mhz,
            fit.fitted_rms_mhz,
            math.nan,
        )
    if not fit.accepted or fit.fitted is None:
        return _attempt(
            trigger,
            False,
            fit.reason,
            len(selected),
            fit.base_rms_mhz,
            fit.fitted_rms_mhz,
            math.nan,
        )
    if candidate is None:
        return _attempt(
            trigger,
            False,
            "overlay predictor unavailable",
            len(selected),
            fit.base_rms_mhz,
            fit.fitted_rms_mhz,
            math.nan,
        )
    if not (math.isfinite(fit.base_rms_mhz) and math.isfinite(fit.fitted_rms_mhz)):
        return _attempt(
            trigger,
            False,
            "fit RMS is not finite",
            len(selected),
            fit.base_rms_mhz,
            fit.fitted_rms_mhz,
            math.nan,
        )
    if fit.fitted_rms_mhz > fit.base_rms_mhz:
        return _attempt(
            trigger,
            False,
            "fit RMS is worse than active base",
            len(selected),
            fit.base_rms_mhz,
            fit.fitted_rms_mhz,
            math.nan,
        )
    if fit.fitted_rms_mhz > cfg.max_rms_mhz:
        return _attempt(
            trigger,
            False,
            "fit RMS exceeds physical_recovery_max_rms_mhz",
            len(selected),
            fit.base_rms_mhz,
            fit.fitted_rms_mhz,
            math.nan,
        )

    try:
        base_center = float(base_predictor.predict_freq(env.flux))
    except Exception:
        base_center = float(snapshot_predict_freq)
    try:
        candidate_center = float(candidate.predict_freq(env.flux))
    except Exception as exc:
        raise RuntimeError(
            "qubit_freq physical recovery candidate prediction failed"
        ) from exc
    center_shift = abs(candidate_center - base_center)
    if not math.isfinite(center_shift):
        return _attempt(
            trigger,
            False,
            "candidate center shift is not finite",
            len(selected),
            fit.base_rms_mhz,
            fit.fitted_rms_mhz,
            center_shift,
        )
    if center_shift > cfg.max_center_shift_mhz:
        return _attempt(
            trigger,
            False,
            "candidate center shift exceeds physical_recovery_max_center_shift_mhz",
            len(selected),
            fit.base_rms_mhz,
            fit.fitted_rms_mhz,
            center_shift,
        )
    if estimator is None:
        return _attempt(
            trigger,
            False,
            "correction estimator unavailable",
            len(selected),
            fit.base_rms_mhz,
            fit.fitted_rms_mhz,
            center_shift,
        )

    residuals = tuple(
        (point.flux, point.frequency_mhz - float(candidate.predict_freq(point.flux)))
        for point in selected
    )
    if any(
        not (math.isfinite(flux) and math.isfinite(residual))
        for flux, residual in residuals
    ):
        return _attempt(
            trigger,
            False,
            "reseed residual is not finite",
            len(selected),
            fit.base_rms_mhz,
            fit.fitted_rms_mhz,
            center_shift,
        )
    estimator.replace_observations(residuals)
    return _attempt(
        trigger,
        True,
        "accepted",
        len(selected),
        fit.base_rms_mhz,
        fit.fitted_rms_mhz,
        center_shift,
    )


def _active_physical_predictor(
    tools: Tools | None,
    state: QubitFreqRecoveryState,
) -> Predictor | None:
    if state.overlay is not None:
        return state.overlay
    if tools is None:
        return None
    return tools.predictor


def _estimator(env: RunEnv, estimator_key: str) -> ScalarEstimator | None:
    if env.feedback is None:
        return None
    return env.feedback.estimator(estimator_key)


def _placement_key(env: RunEnv) -> str:
    return env.node_name or "qubit_freq"


def _greedy_spacing(
    points: tuple[TrustedFrequencyPoint, ...],
    min_spacing: float,
) -> tuple[TrustedFrequencyPoint, ...]:
    selected: list[TrustedFrequencyPoint] = []
    last_flux = -math.inf
    tolerance = max(abs(min_spacing), 1.0) * 1e-12
    for point in points:
        if not selected or point.flux - last_flux >= min_spacing - tolerance:
            selected.append(point)
            last_flux = point.flux
    return tuple(selected)


def _prefer_last_endpoint(
    selected: tuple[TrustedFrequencyPoint, ...],
    last_point: TrustedFrequencyPoint,
    min_spacing: float,
) -> tuple[TrustedFrequencyPoint, ...]:
    if not selected or selected[-1] == last_point:
        return selected
    if len(selected) == 1:
        return (last_point,)
    tolerance = max(abs(min_spacing), 1.0) * 1e-12
    if last_point.flux - selected[-2].flux >= min_spacing - tolerance:
        return (*selected[:-1], last_point)
    return selected


def _thin_evenly(
    points: tuple[TrustedFrequencyPoint, ...],
    max_points: int,
) -> tuple[TrustedFrequencyPoint, ...]:
    if len(points) <= max_points:
        return points
    raw_indices = np.linspace(0, len(points) - 1, max_points)
    indices: list[int] = []
    for raw in raw_indices:
        index = int(round(float(raw)))
        if indices and index <= indices[-1]:
            index = indices[-1] + 1
        indices.append(min(index, len(points) - 1))
    indices[-1] = len(points) - 1
    return tuple(points[index] for index in indices)


def _attempt(
    trigger: str,
    accepted: bool,
    reason: str,
    n_points: int,
    base_rms_mhz: float,
    fitted_rms_mhz: float,
    center_shift_mhz: float,
) -> PhysicalRecoveryAttempt:
    return PhysicalRecoveryAttempt(
        trigger=trigger,
        accepted=accepted,
        reason=reason,
        n_points=int(n_points),
        base_rms_mhz=float(base_rms_mhz),
        fitted_rms_mhz=float(fitted_rms_mhz),
        center_shift_mhz=float(center_shift_mhz),
    )


def _finite(name: str, value: Any) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise RuntimeError(f"qubit_freq {name} must be finite")
    return out


def _positive_finite(name: str, value: Any) -> float:
    out = _finite(name, value)
    if out <= 0.0:
        raise RuntimeError(f"qubit_freq {name} must be positive")
    return out


def _int(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise RuntimeError(f"qubit_freq {name} must be an integer")
    return value
