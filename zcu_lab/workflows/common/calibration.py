"""common calibration helpers for lab workflows."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from zcu_tools.experiment.metadata import RunSnapshot
from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.workflows import Failed
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import SweepCfg
from zcu_tools.program.v2.modules import PulseCfg, PulseReadoutCfg, ReadoutCfg
from zcu_tools.utils.process import SmoothingOptions, smooth_signal_nd

from zcu_lab.v2.twotone.freq.core import (
    FreqCfg,
    FreqExp,
    FreqResult,
)
from zcu_lab.v2.twotone.rabi.len_rabi.core import (
    LenRabiCfg,
    LenRabiExp,
)
from zcu_lab.v2.twotone.ro_optimize.freq_gain.core import (
    FreqGainCfg,
    FreqGainExp,
)
from zcu_lab.workflows.common.calibration_analysis import (
    FrequencyFit as FrequencyFit,
)
from zcu_lab.workflows.common.calibration_analysis import (
    RabiFit as RabiFit,
)
from zcu_lab.workflows.common.calibration_analysis import (
    analyze_len_rabi as analyze_len_rabi,
)
from zcu_lab.workflows.common.calibration_analysis import (
    analyze_qubit_freq as analyze_qubit_freq,
)
from zcu_lab.workflows.common.calibration_plots import (
    CalibrationTrace,
    draw_frequency,
    draw_rabi,
    draw_readout,
    retain_trace,
)
from zcu_lab.workflows.common.calibration_results import (
    FrequencyResult as FrequencyResult,
)
from zcu_lab.workflows.common.calibration_results import (
    RabiResult as RabiResult,
)
from zcu_lab.workflows.common.calibration_results import (
    ReadoutResult as ReadoutResult,
)
from zcu_lab.workflows.common.calibration_results import (
    StageStatus,
)
from zcu_lab.workflows.common.frequency_tracking import (
    FrequencyTracker,
    FrequencyTunables,
    trusted_curve,
)
from zcu_lab.workflows.common.rabi_feedback import RabiDriveFeedback
from zcu_lab.workflows.common.readout_analysis import accepted_readout_optimum
from zcu_lab.workflows.common.saving import save_native


class CalibrationTunables(FrequencyTunables):
    """Generation and acceptance knobs captured once per flux-point invocation.

    qf_widen specifies the finite sequence of spectroscopy spans, not retries
    after adapter errors. Error and residual upper bounds are strict; readout
    SNR uses an inclusive lower bound. ro_every counts flux points.
    drive_gain_mode controls spectroscopy; rabi_drive_gain_mode controls Rabi.
    Acceptance/cadence/range defaults are provisional simulator settings, not
    hardware-validated physics defaults; users confirm them at CP-B/CP-C.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    qf_reps: int = Field(default=100, ge=1, description="Spectroscopy averages")
    qf_rounds: int = Field(default=1, ge=1, description="Spectroscopy rounds")
    qf_acquire_retry: int = Field(
        default=3, ge=0, description="Retries after a transient acquisition failure"
    )
    qf_earlystop_snr: float | None = Field(
        default=50.0,
        gt=0,
        description="Completed-round trace SNR threshold; None disables early stop",
    )
    rabi_reps: int = Field(default=100, ge=1, description="Rabi averages")
    rabi_rounds: int = Field(default=1, ge=1, description="Rabi rounds")
    rabi_acquire_retry: int = Field(
        default=3, ge=0, description="Retries after a transient acquisition failure"
    )
    rabi_earlystop_snr: float | None = Field(
        default=30.0,
        gt=0,
        description="Completed-round trace SNR threshold; None disables early stop",
    )
    ro_reps: int = Field(default=100, ge=1, description="Readout averages")
    ro_rounds: int = Field(default=1, ge=1, description="Readout rounds")
    ro_acquire_retry: int = Field(
        default=3, ge=0, description="Retries after a transient acquisition failure"
    )
    max_frequency_rel_err: float = Field(
        default=0.3, gt=0, le=1, description="Strict center-error/FWHM upper bound"
    )
    max_rabi_rel_err: float = Field(
        default=0.3, gt=0, le=1, description="Strict pi-error/pi-duration upper bound"
    )
    rabi_max_mean_residual_ratio: float = Field(
        default=0.1,
        gt=0,
        description="Strict mean absolute Rabi residual/fit-range upper bound",
    )
    max_fit_residual: float = Field(
        default=0.2,
        gt=0,
        le=1,
        description="Strict normalized residual RMS upper bound",
    )
    min_readout_snr: float = Field(
        default=5.0, gt=0, description="Inclusive minimum readout SNR"
    )
    ro_every: int = Field(
        default=5, ge=1, description="Flux-point readout scan cadence"
    )
    rabi_start_factor: float = Field(
        default=0.1, gt=0, description="Sweep start divided by prior pi duration"
    )
    rabi_stop_factor: float = Field(
        default=3.0, gt=0, description="Sweep stop divided by prior pi duration"
    )

    rabi_drive_gain_mode: Literal["auto_pi_product", "fixed"] = Field(
        default="auto_pi_product",
        description="Predict/control gain or retain template gain",
    )
    target_pi_length_us: float | None = Field(
        default=None,
        gt=0,
        description="Pi-duration setpoint in us; None uses the host pi seed",
    )
    rabi_min_start_us: float = Field(
        default=0.05, ge=0, description="Minimum Rabi sweep start in us"
    )
    rabi_min_stop_us: float = Field(
        default=0.5, gt=0, description="Minimum Rabi sweep stop in us"
    )
    rabi_points: int = Field(default=101, ge=3, description="Rabi sweep point count")
    rabi_relax_mode: Literal["auto", "fixed"] = Field(
        default="auto", description="T1-derived or fixed Rabi relaxation"
    )
    rabi_relax_factor: float = Field(
        default=3.0, gt=0, description="Rabi relaxation divided by smoothed T1"
    )
    rabi_relax_min_us: float = Field(
        default=0.0, ge=0, description="Rabi relaxation floor in us"
    )
    rabi_relax_delay_us: float = Field(
        default=30.0, ge=0, description="Fixed Rabi relaxation in us"
    )
    ro_relax_mode: Literal["auto", "fixed"] = Field(
        default="auto", description="T1-derived or fixed readout relaxation"
    )
    ro_relax_factor: float = Field(
        default=3.0, gt=0, description="Readout relaxation divided by smoothed T1"
    )
    ro_relax_min_us: float = Field(
        default=0.0, ge=0, description="Readout relaxation floor in us"
    )
    ro_relax_delay_us: float = Field(
        default=30.0, ge=0, description="Fixed readout relaxation in us"
    )
    ro_freq_half_width_mhz: float = Field(
        default=1.0, gt=0, description="Readout frequency search half width in MHz"
    )
    ro_gain_half_width: float = Field(
        default=0.1, gt=0, le=1, description="Readout gain search half width"
    )
    ro_skew_penalty: float = Field(
        default=0.0, ge=0, description="Readout IQ skewness score penalty"
    )

    @model_validator(mode="after")
    def _validate_rabi_range(self) -> CalibrationTunables:
        if self.rabi_stop_factor <= self.rabi_start_factor:
            raise ValueError("Rabi stop factor must exceed start factor")
        if self.rabi_min_stop_us <= self.rabi_min_start_us:
            raise ValueError("Rabi stop floor must exceed start floor")
        return self


@dataclass(frozen=True)
class Calibrated:
    """Accepted scalar calibration, also usable as deeply read-only host seed.

    f01_mhz/readout_freq_mhz are positive finite frequencies in MHz.
    pi_length_us is a positive finite pi-pulse duration at drive_gain.
    readout_gain is a finite nonnegative native readout gain.
    drive_gain is the acquisition gain in [0, 1]; None retains the template gain.
    Invalid values raise ValueError at construction.
    """

    f01_mhz: float
    pi_length_us: float
    readout_freq_mhz: float
    readout_gain: float
    drive_gain: float | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        for name, value in (
            ("f01_mhz", self.f01_mhz),
            ("pi_length_us", self.pi_length_us),
            ("readout_freq_mhz", self.readout_freq_mhz),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.readout_gain) or self.readout_gain < 0:
            raise ValueError("readout_gain must be nonnegative and finite")
        if self.drive_gain is not None and (
            not math.isfinite(self.drive_gain) or not 0 <= self.drive_gain <= 1
        ):
            raise ValueError("drive_gain must be finite and in [0, 1]")


@dataclass(frozen=True)
class CalibratedPi2(Calibrated):
    """Successful calibration with a fitted pi/2 duration for Echo and Ramsey.

    The original four Calibrated scalars retain their meanings and validation.
    pi2_length_us is a positive finite, phase-aware Length-Rabi fit duration in
    microseconds at drive_gain and the selected waveform. It is at least 0.03 us.
    It is neither half pi_length_us nor an initial seed; callers obtain it from accepted Rabi.
    Invalid original scalars or pi2 duration raise ValueError at construction.
    """

    pi2_length_us: float

    def __post_init__(self) -> None:
        super().__post_init__()
        if not math.isfinite(self.pi2_length_us) or self.pi2_length_us < 0.03:
            raise ValueError("pi2_length_us must be finite and at least 0.03 us")


@dataclass
class CalCarry:
    """Coroutine-local calibration carry retained between committed points.

    seed is the legal host scalar seed; last contains the latest accepted
    scalar parts or None initially. Frequency, Rabi and readout update only
    their own fields. f01_by_flux contains accepted (device setpoint, MHz)
    frequency history even when a later stage fails. frequency_traces/rabi_traces
    retain one projected sweep row per point for run-local heatmaps, not Figures.
    rabi_drive holds the pi-product history and query-aged gain controller.
    last_pi retains the frequency paired with accepted pi/pi2; pi_source and
    readout_source identify their producer points, or None before acceptance.
    Widening replaces the current point's frequency row. Pause preserves this carry;
    only explicit point commits persist reports. There are no stage checkpoints.
    frequency_tracker owns the run-local physical overlay, IDW and linewidth
    feedback. cur_m is |<0|n|1>| and m_ratio is relative to the first flux point;
    both are None before prediction (ratio also None for a zero reference).
    """

    seed: Calibrated
    last: Calibrated | None = None
    f01_by_flux: list[tuple[float, float]] = field(default_factory=list)
    frequency_traces: list[CalibrationTrace] = field(default_factory=list)
    rabi_traces: list[CalibrationTrace] = field(default_factory=list)
    rabi_drive: RabiDriveFeedback = field(default_factory=RabiDriveFeedback)
    last_pi: CalibratedPi2 | None = None
    pi_source: CalibrationSource | None = None
    readout_source: CalibrationSource | None = None
    frequency_tracker: FrequencyTracker | None = None
    # Matrix-element feed-forward for the Rabi lane; refreshed at every point.
    cur_m: float | None = None
    m_ratio: float | None = None


class CalibrationSource(BaseModel):
    """Identify the accepted dependency used by a consumer at a flux point.

    need is latest (prior accepted points allowed) or now (current point only).
    source_index/source_flux identify the producer; None means the fixed host
    seed. fallback marks a prior-point or seed dependency, not a new acceptance.
    """

    need: Literal["latest", "now"] = "latest"
    source_index: int | None = None
    source_flux: float | None = None
    fallback: bool = True


@dataclass(frozen=True)
class CalibrationInputs:
    """Fixed native routing and captured policy shared by the three experiments.

    qub_pulse/readout are resolved shared drive/pulse-backed readout modules.
    tun is this invocation's ordinary generation/acceptance/cadence policy.
    The wrapper is frozen; nested modules are copied before applying scalars.
    snapshot is the fixed historical entry/point/role state used by all three
    native saves, not a live source. directory is the allocated commit directory.
    Coherence and MIST reuse fixed modules/snapshot/directory across stages,
    replacing tun from tun.current.cal before each stage. A stage does not
    recapture sources or refresh its own policy during acquisition.
    flux/flux_unit label the current point's history rows in device coordinates.
    """

    qub_pulse: PulseCfg
    readout: PulseReadoutCfg
    tun: CalibrationTunables
    snapshot: RunSnapshot
    directory: Path
    point_index: int
    flux: float
    flux_unit: str


def apply_readout_calibration(readout: ReadoutCfg, cal: Calibrated) -> None:
    """Apply accepted readout scalars to a caller-owned native cfg in place.

    Updates the shared generator/ADC frequency in MHz and native readout gain.
    Callers must copy any template they need to retain before calling.
    Native set_param errors propagate. No acquisition or saving occurs.
    """
    readout.set_param("freq", cal.readout_freq_mhz)
    readout.set_param("gain", cal.readout_gain)


def _validate_calibration_inputs(flux: float, t1_estimate_us: float) -> None:
    """Reject nonfinite flux and nonpositive/nonfinite T1 estimates in µs."""
    if not math.isfinite(t1_estimate_us) or t1_estimate_us <= 0:
        raise ValueError("T1 estimate must be positive and finite")
    if not math.isfinite(flux):
        raise ValueError("Calibration flux must be finite")


def _frequency_cfg(
    inputs: CalibrationInputs,
    prior: Calibrated,
    center: float,
    gain: float,
    widen: float,
) -> FreqCfg:
    drive = inputs.qub_pulse.model_copy(deep=True)
    drive.set_param("gain", gain)
    ro = inputs.readout.model_copy(deep=True)
    apply_readout_calibration(ro, prior)
    span = inputs.tun.span_mhz * widen
    offset = inputs.tun.detune_center_mhz
    return FreqCfg(
        reps=inputs.tun.qf_reps,
        rounds=inputs.tun.qf_rounds,
        acquire_retry=inputs.tun.qf_acquire_retry,
        earlystop_snr=inputs.tun.qf_earlystop_snr,
        relax_delay=1.0,
        qub_pulse=drive,
        readout=ro,
        freq=SweepCfg(
            start=center + offset - span / 2,
            stop=center + offset + span / 2,
            expts=inputs.tun.qf_points,
            step=span / (inputs.tun.qf_points - 1),
        ),
    )


def _accept_frequency(
    fit: FrequencyFit, result: FreqResult, tun: CalibrationTunables
) -> bool:
    margin = tun.frequency_edge_margin * float(np.ptp(result.freqs))
    residual = fit.quality.normalized_residual_rms
    return (
        math.isfinite(fit.frequency_mhz)
        and fit.frequency_mhz > 0
        and float(np.min(result.freqs)) + margin
        <= fit.frequency_mhz
        <= float(np.max(result.freqs)) - margin
        and math.isfinite(fit.width_mhz)
        and fit.width_mhz > 0
        and math.isfinite(fit.error_mhz)
        and fit.error_mhz >= 0
        and fit.error_mhz / fit.width_mhz < tun.max_frequency_rel_err
        and residual is not None
        and residual < tun.max_fit_residual
        and trusted_curve(fit.observed, fit.fitted, 0.2)
    )


def _observe_linewidth(
    tracker: FrequencyTracker,
    fit: FrequencyFit,
    result: FreqResult,
    gain: float,
    index: int,
    tun: CalibrationTunables,
) -> None:
    if 0 < fit.width_mhz <= float(np.ptp(result.freqs)) and trusted_curve(
        fit.observed, fit.fitted, tun.linewidth_max_mean_residual_ratio
    ):
        tracker.observe_linewidth(index, fit.width_mhz, gain)


async def calibrate_frequency(
    scope: Scope,
    *,
    plots: Plots,
    inputs: CalibrationInputs,
    flux: float,
    carry: CalCarry,
) -> FrequencyResult:
    """Locate the qubit resonance so this flux point can use a resonant drive.

    Await in a workflow. flux is a finite device setpoint. carry is
    coroutine-local; only frequency/history change. inputs supplies the
    resolved modules, this invocation's policy and the fixed historical snapshot
    used by the native saver after each Completed attempt,
    before fitting. Widening uses inputs.tun.qf_widen; the center follows the
    captured physical model plus confidence-decayed IDW residuals. Probe gain
    follows trusted linewidth feedback. Relax timing is 1 us. Acceptance uses
    both the captured gates and the legacy mean-error frequency gate.
    Partial attempts save then return rejected/stopped without fitting. Widening
    continues as usual. Expected fit failures return failed/reason, with all
    Completed refs retained. Failure advances only recovery state, while partial
    leaves accepted history and feedback observations unchanged.
    Invalid flux raises before acquisition; all unexpected errors propagate.
    """
    # Check the setpoint before acquiring and project the frequency history.
    if not math.isfinite(flux):
        raise ValueError("Calibration flux must be finite")
    prior = carry.seed if carry.last is None else carry.last
    tracker = carry.frequency_tracker
    if tracker is None:
        raise ValueError("Frequency calibration requires a captured physical predictor")
    hint = tracker.begin_point(flux)
    predictor_model = tracker.model_snapshot()
    carry.cur_m, carry.m_ratio = tracker.cur_m, tracker.m_ratio
    gain = tracker.probe_gain(float(inputs.qub_pulse.gain), inputs.tun)
    filenames: list[str] = []
    gains: list[float] = []
    status: StageStatus = "rejected"
    reason = "No accepted frequency fit"
    for attempt, widen in enumerate(inputs.tun.qf_widen, start=1):
        if scope.cancelled:
            return FrequencyResult(
                "skipped",
                "Point cancelled",
                None,
                tuple(filenames),
                tuple(gains),
                hint,
                predictor_model,
                tracker.last_recovery,
            )
        cfg = _frequency_cfg(inputs, prior, hint, gain, widen)
        gains.append(gain)
        outcome = await scope.run(FreqExp().run, cfg, name="frequency")
        if isinstance(outcome, Failed):
            status, reason = "failed", outcome.reason
            continue
        # Save every completed attempt, then fit and accept its resonance.
        filename = f"frequency-{attempt:02d}.h5"
        save_native(
            outcome,
            inputs.directory / filename,
            spec=FreqExp.AXES_SPEC,
            snapshot=inputs.snapshot,
            identity=scope.identity,
        )
        filenames.append(filename)
        if outcome.partial_result:
            status, reason = "rejected", "stopped"
            continue
        result = outcome.value.result
        fit = analyze_qubit_freq(result)
        trace = CalibrationTrace(
            inputs.point_index,
            flux,
            result.freqs - hint,
            result.signals.real if isinstance(fit, Failed) else fit.observed,
        )
        retain_trace(carry.frequency_traces, trace)
        if isinstance(fit, Failed):
            draw_frequency(
                plots,
                carry.frequency_traces,
                carry.f01_by_flux,
                flux_unit=inputs.flux_unit,
                directory=inputs.directory,
            )
            status, reason = "failed", fit.reason
            continue
        accepted = _accept_frequency(fit, result, inputs.tun)
        if accepted:
            carry.last = replace(prior, f01_mhz=fit.frequency_mhz)
            carry.f01_by_flux.append((flux, fit.frequency_mhz))
            tracker.finish_point(flux, fit.frequency_mhz, tun=inputs.tun)
            _observe_linewidth(
                tracker, fit, result, gain, inputs.point_index, inputs.tun
            )
        carry.frequency_traces[-1] = replace(
            trace,
            observed=fit.observed,
            fitted=fit.fitted,
            accepted=fit.frequency_mhz - hint,
        )
        draw_frequency(
            plots,
            carry.frequency_traces,
            carry.f01_by_flux,
            flux_unit=inputs.flux_unit,
            directory=inputs.directory,
        )
        if accepted:
            return FrequencyResult(
                "accepted",
                None,
                fit.frequency_mhz,
                tuple(filenames),
                tuple(gains),
                hint,
                predictor_model,
                tracker.last_recovery,
            )
        status, reason = "rejected", "No accepted frequency fit"
    if reason != "stopped":
        tracker.finish_point(flux, None, tun=inputs.tun)
    return FrequencyResult(
        status,
        reason,
        None,
        tuple(filenames),
        tuple(gains),
        hint,
        predictor_model,
        tracker.last_recovery,
    )


async def calibrate_rabi(
    scope: Scope,
    *,
    plots: Plots,
    inputs: CalibrationInputs,
    frequency: FrequencyResult,
    carry: CalCarry,
    t1_estimate_us: float,
) -> RabiResult:
    """Measure Rabi oscillations to calibrate pi and pi/2 pulse durations here.

    frequency must be the current point's result, never a prior frequency.
    Missing acceptance returns a named skipped result. t1_estimate_us is positive
    finite historical smoothed us. Relaxation uses the Rabi auto/fixed knobs.
    inputs supplies captured modules/policy. Auto gain blends pi-product prediction
    with confidence-aged log feedback; fixed mode retains template gain. Sweep
    endpoints honor physical floors. Only accepted fits update pi/product feedback.
    Quality-trusted fits below .03 us remain rejected but propose a lower auto
    gain, without changing the usable pulse. inputs.snapshot is the fixed historical entry/point state
    used to save rabi.h5 after Completed, including rejected fits. point_index
    must be a nonnegative integer (not bool), the zero-based flux index; invalid
    indexes raise ValueError at entry. The title shows its flux setpoint. Lengths
    stay in microseconds and the projected signal has arbitrary units.
    Partial saves then returns rejected/stopped without fitting or carry updates.
    Schedule and expected fit failures are named failed; unexpected errors
    propagate. Invalid timing raises before any acquisition. Await in a workflow.
    """
    # Require a valid plot index and this point's accepted frequency.
    if type(inputs.point_index) is not int or inputs.point_index < 0:
        raise ValueError("Rabi point index must be a nonnegative integer")
    _validate_calibration_inputs(0.0, t1_estimate_us)
    if frequency.status != "accepted":
        return RabiResult(
            "skipped",
            "Requires this point's accepted frequency",
            None,
            None,
            None,
            None,
        )
    if frequency.frequency_mhz is None:
        raise ValueError("Accepted frequency result requires MHz")
    f01 = frequency.frequency_mhz
    prior = carry.seed if carry.last is None else carry.last
    # Sweep duration at the accepted frequency, retaining legal readout.
    drive = inputs.qub_pulse.model_copy(deep=True)
    drive.set_param("freq", f01)
    gain = float(drive.gain)
    target_pi_us = (
        carry.seed.pi_length_us
        if inputs.tun.target_pi_length_us is None
        else inputs.tun.target_pi_length_us
    )
    if inputs.tun.rabi_drive_gain_mode == "auto_pi_product":
        gain = carry.rabi_drive.select_gain(
            carry.seed.pi_length_us * float(inputs.qub_pulse.gain),
            target_pi_us,
        )
        drive.set_param("gain", gain)
    ro = inputs.readout.model_copy(deep=True)
    apply_readout_calibration(ro, prior)
    start = max(
        inputs.tun.rabi_min_start_us, inputs.tun.rabi_start_factor * prior.pi_length_us
    )
    stop = max(
        inputs.tun.rabi_min_stop_us, inputs.tun.rabi_stop_factor * prior.pi_length_us
    )
    cfg = LenRabiCfg(
        reps=inputs.tun.rabi_reps,
        rounds=inputs.tun.rabi_rounds,
        acquire_retry=inputs.tun.rabi_acquire_retry,
        earlystop_snr=inputs.tun.rabi_earlystop_snr,
        relax_delay=(
            max(
                inputs.tun.rabi_relax_min_us,
                inputs.tun.rabi_relax_factor * t1_estimate_us,
            )
            if inputs.tun.rabi_relax_mode == "auto"
            else inputs.tun.rabi_relax_delay_us
        ),
        qub_pulse=drive,
        readout=ro,
        length=SweepCfg(
            start=start,
            stop=stop,
            expts=inputs.tun.rabi_points,
            step=(stop - start) / (inputs.tun.rabi_points - 1),
        ),
    )
    outcome = await scope.run(LenRabiExp().run, cfg, name="Rabi")
    if isinstance(outcome, Failed):
        return RabiResult("failed", outcome.reason, f01, None, None, None, gain)
    # Save before fitting, so rejected fits retain their native trace.
    save_native(
        outcome,
        inputs.directory / "rabi.h5",
        spec=LenRabiExp.AXES_SPEC,
        snapshot=inputs.snapshot,
        identity=scope.identity,
    )
    if outcome.partial_result:
        return RabiResult("rejected", "stopped", f01, None, None, "rabi.h5", gain)
    result = outcome.value.result
    fit = analyze_len_rabi(result)
    trace = CalibrationTrace(
        inputs.point_index,
        inputs.flux,
        result.lengths,
        result.signals.real if isinstance(fit, Failed) else fit.observed,
    )
    retain_trace(carry.rabi_traces, trace)
    if isinstance(fit, Failed):
        draw_rabi(
            plots,
            carry.rabi_traces,
            flux_unit=inputs.flux_unit,
            directory=inputs.directory,
        )
        return RabiResult("failed", fit.reason, f01, None, None, "rabi.h5", gain)
    residual = fit.quality.normalized_residual_rms
    quality_trusted = (
        math.isfinite(fit.pi_length_us)
        and 0 < fit.pi_length_us <= 0.9 * float(np.max(result.lengths))
        and trusted_curve(
            fit.observed, fit.fitted, inputs.tun.rabi_max_mean_residual_ratio
        )
        and math.isfinite(fit.error_us)
        and fit.error_us >= 0
        and fit.error_us / fit.pi_length_us < inputs.tun.max_rabi_rel_err
        and residual is not None
        and residual < inputs.tun.max_fit_residual
    )
    too_fast = quality_trusted and fit.pi_length_us < 0.03
    accepted = quality_trusted and not too_fast
    carry.rabi_traces[-1] = replace(
        trace,
        observed=fit.observed,
        fitted=fit.fitted,
        accepted=fit.pi_length_us if accepted else None,
    )
    draw_rabi(
        plots, carry.rabi_traces, flux_unit=inputs.flux_unit, directory=inputs.directory
    )
    if not accepted:
        if too_fast and inputs.tun.rabi_drive_gain_mode == "auto_pi_product":
            carry.rabi_drive.observe_too_fast(
                fit.pi_length_us, gain, max(target_pi_us, 0.03)
            )
        return RabiResult(
            "rejected",
            "π shorter than minimum trusted length"
            if too_fast
            else "Rabi fit rejected",
            f01,
            None,
            None,
            "rabi.h5",
            gain,
            too_fast_pi_length_us=fit.pi_length_us if too_fast else None,
        )
    carry.rabi_drive.observe(
        inputs.point_index,
        fit.pi_length_us,
        gain,
        target_pi_us,
        control=inputs.tun.rabi_drive_gain_mode == "auto_pi_product",
    )
    pi2 = (
        fit.pi2_length_us
        if math.isfinite(fit.pi2_length_us) and fit.pi2_length_us >= 0.03
        else None
    )
    carry.last = Calibrated(
        f01,
        fit.pi_length_us,
        prior.readout_freq_mhz,
        prior.readout_gain,
        drive_gain=gain,
    )
    return RabiResult("accepted", None, f01, fit.pi_length_us, pi2, "rabi.h5", gain)


def readout_due(point_index: int, tun: CalibrationTunables) -> bool:
    """Return whether this zero-based flux point scans readout.

    The rule is point_index % tun.ro_every == 0; index zero naturally scans.
    Failed/rejected prior scans do not change the cadence or cause extra scans.
    The caller retains legal seed/prior readout on cadence skips and checks this
    point's latest-pi dependency separately. Negative indexes raise ValueError.
    """
    if point_index < 0:
        raise ValueError("Readout point index must be nonnegative")
    return point_index % tun.ro_every == 0


async def calibrate_readout(
    scope: Scope,
    *,
    plots: Plots,
    inputs: CalibrationInputs,
    rabi: RabiResult,
    carry: CalCarry,
    t1_estimate_us: float,
) -> ReadoutResult:
    """Scan readout frequency and gain to maximize discrimination after Rabi.

    Resolve the latest accepted pi pulse with its source frequency and gain.
    A failed Rabi can reuse that pulse; no accepted pulse gives a dependency skip.
    Caller uses readout_due to decide cadence. Accepted pulse pairs record their
    producer in carry. Relaxation follows smoothed T1 or the fixed readout knob.
    Search windows follow retained readout with configurable half widths and gain
    clipped to [0, 1]. A wavelet-smoothed map needs a unique interior prominent
    maximum meeting the SNR floor. Failure/rejection retains legal prior
    readout, without returning fabricated accepted values. inputs.snapshot is
    the fixed historical entry/point state used to save readout.h5 after Completed,
    including rejection. Partial saves then returns rejected/stopped without
    selecting a best point or updating carry. Invalid input/unexpected errors raise.
    Await in a workflow; cancellation remains Engine-owned.
    """
    # Readout consumes the latest accepted pi module, unlike Rabi's NOW frequency.
    _validate_calibration_inputs(0.0, t1_estimate_us)
    cal = current_point_calibration(
        rabi, carry, point_index=inputs.point_index, flux=inputs.flux
    )
    if cal is None:
        return ReadoutResult(
            "skipped", "Requires a latest accepted pi pulse", None, None, None
        )
    prior = carry.seed if carry.last is None else carry.last
    # Sweep readout gain and frequency with the calibrated pi drive.
    drive = inputs.qub_pulse.model_copy(deep=True)
    drive.set_param("freq", cal.f01_mhz)
    drive.set_param("length", cal.pi_length_us)
    if cal.drive_gain is not None:
        drive.set_param("gain", cal.drive_gain)
    ro = inputs.readout.model_copy(deep=True)
    apply_readout_calibration(ro, prior)
    gain_start = max(0.0, prior.readout_gain - inputs.tun.ro_gain_half_width)
    gain_stop = min(1.0, prior.readout_gain + inputs.tun.ro_gain_half_width)
    cfg = FreqGainCfg(
        reps=inputs.tun.ro_reps,
        rounds=inputs.tun.ro_rounds,
        acquire_retry=inputs.tun.ro_acquire_retry,
        relax_delay=(
            max(inputs.tun.ro_relax_min_us, inputs.tun.ro_relax_factor * t1_estimate_us)
            if inputs.tun.ro_relax_mode == "auto"
            else inputs.tun.ro_relax_delay_us
        ),
        skew_penalty=inputs.tun.ro_skew_penalty,
        qub_pulse=drive,
        readout=ro,
        freq=SweepCfg(
            start=prior.readout_freq_mhz - inputs.tun.ro_freq_half_width_mhz,
            stop=prior.readout_freq_mhz + inputs.tun.ro_freq_half_width_mhz,
            expts=51,
            step=2 * inputs.tun.ro_freq_half_width_mhz / 50,
        ),
        gain=SweepCfg(
            start=gain_start,
            stop=gain_stop,
            expts=31,
            step=(gain_stop - gain_start) / 30,
        ),
    )
    outcome = await scope.run(FreqGainExp().run, cfg, name="readout")
    if isinstance(outcome, Failed):
        return ReadoutResult("failed", outcome.reason, None, None, None)
    # Save the landscape and choose its best point meeting the SNR threshold.
    save_native(
        outcome,
        inputs.directory / "readout.h5",
        spec=FreqGainExp.AXES_SPEC,
        snapshot=inputs.snapshot,
        identity=scope.identity,
    )
    if outcome.partial_result:
        return ReadoutResult("rejected", "stopped", None, None, "readout.h5")
    signals = np.abs(
        smooth_signal_nd(
            outcome.value.result.signals,
            options=SmoothingOptions(method="wavelet", sigma=1.0),
        )
    )
    draw_readout(
        plots,
        outcome.value.result.freqs,
        outcome.value.result.gains,
        signals,
        inputs_flux=(inputs.flux, inputs.flux_unit),
        directory=inputs.directory,
    )
    optimum = accepted_readout_optimum(
        signals, outcome.value.result.freqs, outcome.value.result.gains
    )
    if (
        optimum is None
        or signals[optimum.freq_index, optimum.gain_index] < inputs.tun.min_readout_snr
    ):
        return ReadoutResult(
            "rejected",
            "Readout landscape has no trusted interior SNR optimum",
            None,
            None,
            "readout.h5",
        )
    frequency = optimum.freq
    gain = optimum.gain
    carry.last = replace(prior, readout_freq_mhz=frequency, readout_gain=gain)
    carry.readout_source = CalibrationSource(
        source_index=inputs.point_index, source_flux=inputs.flux, fallback=False
    )
    return ReadoutResult("accepted", None, frequency, gain, "readout.h5")


def current_pi_calibration(
    rabi: RabiResult | None, carry: CalCarry
) -> Calibrated | None:
    """Return this point's accepted pi with its own frequency/gain, without pi2.

    Nonaccepted or absent Rabi returns None; no prior pi is borrowed. An accepted
    record missing its frequency or pi duration raises ValueError. Readout uses
    its independent latest value or host seed. This resolver does not change the
    retained pi/pi2 pair or its source, so T1/GE NOW cannot overwrite T2 LATEST.
    """
    if rabi is None or rabi.status != "accepted":
        return None
    if rabi.frequency_mhz is None or rabi.pi_length_us is None:
        raise ValueError("Accepted Rabi requires current frequency and pi")
    prior = carry.seed if carry.last is None else carry.last
    return Calibrated(
        rabi.frequency_mhz,
        rabi.pi_length_us,
        prior.readout_freq_mhz,
        prior.readout_gain,
        drive_gain=rabi.drive_gain,
    )


def current_point_calibration(
    rabi: RabiResult,
    carry: CalCarry,
    *,
    point_index: int,
    flux: float,
    need: Literal["latest", "now"] = "latest",
) -> CalibratedPi2 | None:
    """Resolve accepted pulse pairs with latest or current-point-only semantics.

    A trusted current pi/pi2 pair stores its frequency, acquisition gain and
    producer index/flux in carry. Rejected/skipped Rabi or untrusted pi2 leaves
    the previous pair intact. latest may reuse it without an age limit; now
    returns None when this point has no trusted pair. Later frequency/gain
    updates never retune an older pi module. Readout independently uses its
    latest accepted value or host seed. No seed pi/2 is fabricated.
    """
    prior = carry.seed if carry.last is None else carry.last
    if rabi.status == "accepted":
        if rabi.frequency_mhz is None or rabi.pi_length_us is None:
            raise ValueError("Accepted Rabi requires current frequency and pi")
        if rabi.pi2_length_us is not None:
            carry.last_pi = CalibratedPi2(
                rabi.frequency_mhz,
                rabi.pi_length_us,
                prior.readout_freq_mhz,
                prior.readout_gain,
                rabi.pi2_length_us,
                drive_gain=rabi.drive_gain,
            )
            carry.pi_source = CalibrationSource(
                source_index=point_index, source_flux=flux, fallback=False
            )
    if need == "now" and (rabi.status != "accepted" or rabi.pi2_length_us is None):
        return None
    if carry.last_pi is None:
        return None
    return replace(
        carry.last_pi,
        readout_freq_mhz=prior.readout_freq_mhz,
        readout_gain=prior.readout_gain,
    )


def calibration_source(
    source: CalibrationSource | None, point_index: int
) -> CalibrationSource:
    """Mark whether a dependency came from this point, a prior point or host seed."""
    source = CalibrationSource() if source is None else source
    return source.model_copy(update={"fallback": source.source_index != point_index})
