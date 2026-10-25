"""Run-local physics, residual and linewidth feedback for flux spectroscopy."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Annotated, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator
from zcu_tools.simulate.fluxonium.physical_fit import (
    FluxoniumLocalFitResult,
    FluxoniumModelSnapshot,
    fit_local_fluxonium_model,
)
from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor

from zcu_lab.workflows.common.feedback import IdwEstimator


class FrequencyModel(BaseModel):
    """Start-captured fluxonium energies (GHz) and native device coordinates.

    flux_half/flux_period define the qubit's own DC line. flux_bias is an
    additive device offset, initially zero unless explicitly supplied. Recovery
    changes only a run-local predictor, never this plan or the captured point.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    EJ: float = Field(gt=0, description="Josephson energy in GHz")
    EC: float = Field(gt=0, description="Charging energy in GHz")
    EL: float = Field(gt=0, description="Inductive energy in GHz")
    flux_half: float = Field(description="Half-flux device value")
    flux_period: float = Field(description="Signed device span per flux quantum")
    flux_bias: float = Field(default=0.0, description="Additive device offset")

    @model_validator(mode="after")
    def _nonzero_period(self) -> FrequencyModel:
        if self.flux_period == 0:
            raise ValueError("Flux period must be nonzero")
        return self

    def make_predictor(self) -> FluxoniumPredictor:
        """Build an independent predictor; energies use GHz, queries return MHz."""
        return FluxoniumPredictor(
            (self.EJ, self.EC, self.EL),
            self.flux_half,
            self.flux_period,
            self.flux_bias,
        )


class FrequencyTunables(BaseModel):
    """Spectroscopy window, probe gain and recovery policy for one invocation.

    detune_center_mhz offsets the window relative to the physical prediction;
    span_mhz*qf_widen gives its full width. qf_points includes both endpoints.
    Adaptive gain targets target_kappa using trusted linewidth/gain feedback,
    capped at drive_gain_cap. Fixed mode retains the selected pulse's gain.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    span_mhz: float = Field(default=30.0, gt=0, description="Base spectroscopy span")
    detune_center_mhz: float = Field(
        default=0.0, description="Detune window offset (MHz)"
    )
    qf_points: int = Field(default=201, ge=2, description="Spectroscopy sample count")
    qf_widen: tuple[Annotated[float, Field(gt=0)], ...] = Field(
        default=(1.0, 2.0, 4.0),
        min_length=1,
        description="Ordered spectroscopy span multipliers",
    )
    frequency_edge_margin: float = Field(
        default=0.1,
        ge=0,
        lt=0.5,
        description="Minimum fitted-center distance from either edge as a scan-span fraction",
    )
    linewidth_max_mean_residual_ratio: float = Field(
        default=0.1,
        gt=0,
        description="Strict mean absolute residual/fit-range bound for linewidth feedback",
    )
    drive_gain_mode: Literal["adaptive", "fixed"] = "adaptive"
    target_kappa: float = Field(default=6.5, gt=0, description="Target linewidth (MHz)")
    drive_gain_cap: float = Field(
        default=1.0, gt=0, le=1, description="Maximum adaptive probe gain"
    )
    physical_recovery_mode: Literal["off", "fail_triggered_fit"] = "fail_triggered_fit"


@dataclass(frozen=True)
class RecoveryAttempt:
    """Last recovery diagnostic: trigger, verdict, sample count and MHz errors.

    trigger is first_fail or first_success_after_fail. n_points counts the
    selected fit points, or zero when fewer than ten trusted points exist.
    base/fitted hold EJ, EC, EL (GHz), device affine map and flux bias before
    and after fitting; fitted is None if no candidate exists. RMS values and
    center_shift_mhz are None when unavailable, never evidence of success.
    accepted means the candidate passed all gates and replaced the predictor.
    """

    trigger: Literal["first_fail", "first_success_after_fail"]
    accepted: bool
    reason: str
    n_points: int
    base: FluxoniumModelSnapshot
    fitted: FluxoniumModelSnapshot | None = None
    base_rms_mhz: float | None = None
    fitted_rms_mhz: float | None = None
    center_shift_mhz: float | None = None


def select_fit_points(
    history: list[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    """Select 10–30 distinct trusted flux points with the old greedy spacing rule.

    Repeated flux values keep the latest measurement. Binary search maximizes
    spacing, retaining the last flux endpoint. Fewer than ten returns no points.
    """
    points = tuple(sorted(dict(history).items()))
    if len(points) < 10:
        return ()
    if len(points) <= 30:
        return points
    low, high = 0.0, (points[-1][0] - points[0][0]) / 29
    while len(_greedy_spacing(points, high)) > 30:
        high *= 2
    best = _greedy_spacing(points, high)
    for _ in range(64):
        mid = (low + high) / 2
        candidate = _greedy_spacing(points, mid)
        if len(candidate) > 30:
            low = mid
        else:
            best, high = candidate, mid
    if not best:
        return ()
    tolerance = max(abs(high), 1.0) * 1e-12
    if best[-1] != points[-1] and (
        len(best) == 1 or points[-1][0] - best[-2][0] >= high - tolerance
    ):
        best = (*best[:-1], points[-1])
    if len(best) > 30:
        indices = np.rint(np.linspace(0, len(best) - 1, 30)).astype(int)
        best = tuple(best[index] for index in indices)
    return best if len(best) >= 10 else ()


def _greedy_spacing(
    points: tuple[tuple[float, float], ...], spacing: float
) -> tuple[tuple[float, float], ...]:
    selected: list[tuple[float, float]] = []
    tolerance = max(abs(spacing), 1.0) * 1e-12
    for point in points:
        if not selected or point[0] - selected[-1][0] >= spacing - tolerance:
            selected.append(point)
    return tuple(selected)


def trusted_curve(
    observed: NDArray[np.float64], fitted: NDArray[np.float64], threshold: float
) -> bool:
    """Accept mean absolute fit error strictly below threshold times fit range."""
    return bool(np.mean(np.abs(observed - fitted)) < threshold * np.ptp(fitted))


@dataclass
class FrequencyTracker:
    """Track physical f01, IDW residuals and step-weighted linewidth per run.

    predictor is an independent model from the fixed plan. begin_point queries
    residual confidence once (not once per widening), and publishes cur_m and
    m_ratio relative to the first point. finish_point observes trusted frequency
    or counts a failed point; observe_linewidth updates gain feedback. Recovery runs on the
    first failure and first success after failure, activating only a validated
    overlay and reseeding residuals against that overlay.
    """

    predictor: FluxoniumPredictor
    residuals: IdwEstimator = field(
        default_factory=lambda: IdwEstimator(k=10, epsilon=1e-4, decay_points=4.0)
    )
    history: list[tuple[float, float]] = field(default_factory=list)
    fail_streak: int = 0
    # Current point only: begin_point clears this so later records cannot repeat it.
    last_recovery: RecoveryAttempt | None = None
    cur_m: float | None = None
    m_ratio: float | None = None
    first_m: float | None = None
    qfw_factor: float | None = None
    last_linewidth_index: int = 0

    def model_snapshot(self) -> FluxoniumModelSnapshot:
        """Capture the current EJ/EC/EL (GHz), DC affine map and flux bias."""
        return FluxoniumModelSnapshot(
            self.predictor.params,
            self.predictor.flux_half,
            self.predictor.flux_period,
            self.predictor.flux_bias,
        )

    def begin_point(self, flux: float) -> float:
        """Clear point recovery; return corrected MHz center and publish matrix element."""
        self.last_recovery = None
        base = float(self.predictor.predict_freq(flux))
        self.cur_m = float(self.predictor.predict_matrix_element(flux))
        if self.first_m is None:
            self.first_m = self.cur_m
        self.m_ratio = self.cur_m / self.first_m if self.first_m > 0 else None
        correction = self.residuals.estimate(flux)
        center = base + (
            correction.confidence * correction.value if correction else 0.0
        )
        if not math.isfinite(center) or center <= 0:
            raise ValueError(
                "Predicted spectroscopy center must be positive and finite"
            )
        return center

    def probe_gain(self, initial_gain: float, tun: FrequencyTunables) -> float:
        """Return initial/fixed gain, or target linewidth divided by smoothed factor."""
        if tun.drive_gain_mode == "fixed" or self.qfw_factor is None:
            return initial_gain
        return min(tun.drive_gain_cap, tun.target_kappa / self.qfw_factor)

    def finish_point(
        self,
        flux: float,
        frequency_mhz: float | None,
        *,
        tun: FrequencyTunables,
    ) -> None:
        """Observe accepted MHz or one exhausted failure; possibly refit EJ/EC/EL/bias."""
        if frequency_mhz is None:
            self.fail_streak += 1
            if self.fail_streak == 1 and tun.physical_recovery_mode != "off":
                self._recover(flux, "first_fail")
            return
        self.history.append((flux, frequency_mhz))
        reseeded = False
        if self.fail_streak > 0 and tun.physical_recovery_mode != "off":
            reseeded = self._recover(flux, "first_success_after_fail")
        self.fail_streak = 0
        if not reseeded:
            self.residuals.observe(
                flux, frequency_mhz - float(self.predictor.predict_freq(flux))
            )

    def observe_linewidth(self, index: int, width_mhz: float, gain: float) -> None:
        """Smooth trusted FWHM/gain with previous weight 0.7**gap; require gain > 0."""
        if gain <= 0:
            raise ValueError("Trusted linewidth requires a positive probe gain")
        factor = width_mhz / gain
        if self.qfw_factor is not None:
            weight = 0.7 ** max(1, index - self.last_linewidth_index)
            factor = (1 - weight) * factor + weight * self.qfw_factor
        self.qfw_factor = factor
        self.last_linewidth_index = index

    def _recover(
        self, flux: float, trigger: Literal["first_fail", "first_success_after_fail"]
    ) -> bool:
        base = self.model_snapshot()
        points = select_fit_points(self.history)
        if not points:
            self.last_recovery = RecoveryAttempt(
                trigger, False, "not enough trusted history", 0, base
            )
            return False
        fit = fit_local_fluxonium_model(base, points)
        candidate = fit.predictor
        shift = math.nan
        reason = fit.reason
        if fit.fitted is not None and candidate is not None:
            shift = abs(
                float(candidate.predict_freq(flux))
                - float(self.predictor.predict_freq(flux))
            )
            if fit.accepted:
                reason = _recovery_rejection(fit, shift)
        residuals = (
            [(x, y - float(candidate.predict_freq(x))) for x, y in points]
            if reason == "accepted" and candidate is not None
            else []
        )
        if any(not math.isfinite(value) for _, value in residuals):
            reason = "reseed residual is not finite"
        accepted = reason == "accepted" and candidate is not None
        self.last_recovery = RecoveryAttempt(
            trigger,
            accepted,
            reason,
            len(points),
            fit.base,
            fit.fitted,
            fit.base_rms_mhz if math.isfinite(fit.base_rms_mhz) else None,
            fit.fitted_rms_mhz if math.isfinite(fit.fitted_rms_mhz) else None,
            shift if math.isfinite(shift) else None,
        )
        if not accepted or candidate is None:
            return False
        self.residuals.replace_observations(residuals)
        self.predictor = candidate
        return True


def _recovery_rejection(fit: FluxoniumLocalFitResult, shift: float) -> str:
    if not (math.isfinite(fit.base_rms_mhz) and math.isfinite(fit.fitted_rms_mhz)):
        return "fit RMS is not finite"
    if fit.fitted_rms_mhz > fit.base_rms_mhz:
        return "fit RMS is worse than active base"
    if fit.fitted_rms_mhz > 50.0:
        return "fit RMS exceeds 50 MHz"
    if not math.isfinite(shift) or shift > 150.0:
        return "candidate center shift exceeds 150 MHz or is not finite"
    return "accepted"
