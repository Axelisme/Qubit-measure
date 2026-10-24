"""common calibration helpers for lab workflows."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator
from zcu_tools.analysis.fitting import (
    FitQuality,
    compute_fit_quality,
    fit_qubit_freq,
    fit_rabi,
)
from zcu_tools.experiment.metadata import RunSnapshot
from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.workflows import Failed
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import SweepCfg
from zcu_tools.program.v2.modules import PulseCfg, PulseReadoutCfg, ReadoutCfg

from zcu_lab.v2.twotone.freq.core import (
    FreqCfg,
    FreqExp,
    FreqResult,
    qubfreq_signal2real,
)
from zcu_lab.v2.twotone.rabi.len_rabi.core import (
    LenRabiCfg,
    LenRabiExp,
    LenRabiResult,
    rabi_signal2real,
)
from zcu_lab.v2.twotone.ro_optimize.freq_gain.core import (
    FreqGainCfg,
    FreqGainExp,
)
from zcu_lab.workflows.common.analysis import (
    fit_or_failed,
    validate_fit_trace,
)
from zcu_lab.workflows.common.calibration_plots import (
    CalibrationTrace,
    draw_frequency,
    draw_rabi,
    draw_readout,
    retain_trace,
)
from zcu_lab.workflows.common.frequency_tracking import (
    FrequencyTracker,
    FrequencyTunables,
    trusted_curve,
)
from zcu_lab.workflows.common.saving import save_native


@dataclass(frozen=True)
class FrequencyFit:
    """Lorentzian frequency estimate and same-coordinate projected curves.

    frequency_mhz/error_mhz are the fitted center and its marginal standard
    error. width_mhz/width_error_mhz are the FWHM and its standard error.
    observed/fitted are float64 vectors aligned with source.freqs.
    quality contains residual diagnostics with no parameter-error entries.
    Values are not acceptance-filtered: native nonfinite errors and unavailable
    diagnostics remain visible for the workflow to reject.
    """

    frequency_mhz: float
    error_mhz: float
    width_mhz: float
    width_error_mhz: float
    observed: NDArray[np.float64]
    fitted: NDArray[np.float64]
    quality: FitQuality


@dataclass(frozen=True)
class RabiFit:
    """Nondecaying Rabi fit with native phase estimation.

    pi_length_us/error_us are the fitted pi-pulse duration and its propagated
    standard error. pi2_length_us/pi2_error_us retain the phase-aware native
    pi/2 duration and propagated error, not half the fitted pi duration.
    Both durations use the same acquisition waveform and gain, in microseconds.
    observed/fitted are float64 vectors aligned with
    source.lengths. quality contains residual diagnostics with no
    parameter-error entries. Native nonfinite estimates/errors are retained;
    this value does not certify calibration acceptance.
    """

    pi_length_us: float
    error_us: float
    pi2_length_us: float
    pi2_error_us: float
    observed: NDArray[np.float64]
    fitted: NDArray[np.float64]
    quality: FitQuality


def analyze_qubit_freq(result: FreqResult) -> FrequencyFit | Failed:
    """Fit the spectroscopy trace to estimate the qubit resonance and linewidth.

    result uses MHz coordinates and complex128 signals of matching shape.
    Reuse the native background-removal/magnitude projection without skipping
    or masking observations. Return estimate, error, FWHM and residual curves.
    Do not mutate result or acquire, plot, save, or apply an acceptance threshold.
    Return Failed(reason) for nonfinite measured IQ or the expected fit
    failures documented by fit_or_failed. Structural/configuration errors and
    unexpected projection/fitter errors propagate; no samples are replaced.
    """
    failure = validate_fit_trace(result.freqs, result.signals)
    if failure is not None:
        return failure
    # Remove the native background, then fit the complete resonance trace.
    observed = qubfreq_signal2real(result.signals)
    fit = fit_or_failed(lambda: fit_qubit_freq(result.freqs, observed, type="lor"))
    if isinstance(fit, Failed):
        return fit
    frequency, error, width, width_error, fitted, _optimizer = fit
    # The workflow uses public estimates/errors, not positional optimizer vectors.
    quality = compute_fit_quality(observed, fitted, {}, None)
    return FrequencyFit(
        float(frequency),
        float(error),
        float(width),
        float(width_error),
        observed,
        fitted,
        quality,
    )


def analyze_len_rabi(result: LenRabiResult) -> RabiFit | Failed:
    """Fit Rabi oscillations to estimate phase-aware pi and pi/2 durations.

    result uses microsecond coordinates and complex128 signals of matching
    1-D shape. Reuse the native real-axis rotation and phase-aware pi estimate.
    Estimating phase avoids a fixed-phase optimizer local minimum on native
    traces whose first sample is nonzero.
    Return pi and pi/2 durations, propagated uncertainties and residual curves.
    Do not skip/mask data, mutate result or acquire, plot, save, or accept a fit.
    Nonfinite measured IQ and fit_or_failed's expected fit failures return
    Failed(reason). Structural/configuration and unexpected errors propagate.
    """
    failure = validate_fit_trace(result.lengths, result.signals)
    if failure is not None:
        return failure
    # Rotate IQ to the signal axis, then fit oscillations with free phase.
    observed = rabi_signal2real(result.signals)
    fit = fit_or_failed(
        lambda: fit_rabi(result.lengths, observed, decay=False, init_phase=None)
    )
    if isinstance(fit, Failed):
        return fit
    pi_length, error, pi2, pi2_error, _freq, _freq_error, fitted, _optimizer = fit
    quality = compute_fit_quality(observed, fitted, {}, None)
    return RabiFit(
        float(pi_length),
        float(error),
        float(pi2),
        float(pi2_error),
        observed,
        fitted,
        quality,
    )


class CalibrationTunables(FrequencyTunables):
    """Generation and acceptance knobs captured once per flux-point invocation.

    qf_widen specifies the finite sequence of spectroscopy spans, not retries
    after adapter errors. Error and residual upper bounds are strict; readout
    SNR uses an inclusive lower bound. ro_every counts flux points.
    Acceptance/cadence/range defaults are provisional simulator settings, not
    hardware-validated physics defaults; users confirm them at CP-B/CP-C.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    qf_reps: int = Field(default=100, ge=1, description="Spectroscopy averages")
    qf_rounds: int = Field(default=1, ge=1, description="Spectroscopy rounds")
    rabi_reps: int = Field(default=100, ge=1, description="Rabi averages")
    rabi_rounds: int = Field(default=1, ge=1, description="Rabi rounds")
    ro_reps: int = Field(default=100, ge=1, description="Readout averages")
    ro_rounds: int = Field(default=1, ge=1, description="Readout rounds")
    max_frequency_rel_err: float = Field(
        default=0.3, gt=0, le=1, description="Strict center-error/FWHM upper bound"
    )
    max_rabi_rel_err: float = Field(
        default=0.3, gt=0, le=1, description="Strict pi-error/pi-duration upper bound"
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

    @model_validator(mode="after")
    def _validate_rabi_range(self) -> CalibrationTunables:
        if self.rabi_stop_factor <= self.rabi_start_factor:
            raise ValueError("Rabi stop factor must exceed start factor")
        return self


@dataclass(frozen=True)
class Calibrated:
    """Accepted scalar calibration, also usable as deeply read-only host seed.

    f01_mhz/readout_freq_mhz are positive finite frequencies in MHz.
    pi_length_us is a positive finite pi-pulse duration at the template gain.
    readout_gain is a finite nonnegative native readout gain.
    Invalid values raise ValueError at construction.
    """

    f01_mhz: float
    pi_length_us: float
    readout_freq_mhz: float
    readout_gain: float

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


@dataclass(frozen=True)
class CalibratedPi2(Calibrated):
    """Successful calibration with a fitted pi/2 duration for Echo and Ramsey.

    The original four Calibrated scalars retain their meanings and validation.
    pi2_length_us is a positive finite, phase-aware Length-Rabi fit duration in
    microseconds at the selected template's gain and waveform. It is neither
    half pi_length_us nor an initial seed; callers obtain it from accepted Rabi.
    Invalid original scalars or pi2 duration raise ValueError at construction.
    """

    pi2_length_us: float

    def __post_init__(self) -> None:
        super().__post_init__()
        if not math.isfinite(self.pi2_length_us) or self.pi2_length_us <= 0:
            raise ValueError("pi2_length_us must be positive and finite")


@dataclass
class CalCarry:
    """Coroutine-local calibration carry retained between committed points.

    seed is the legal host scalar seed; last contains the latest accepted
    scalar parts or None initially. Frequency, Rabi and readout update only
    their own fields. f01_by_flux contains accepted (device setpoint, MHz)
    frequency history even when a later stage fails. frequency_traces/rabi_traces
    retain one projected sweep row per point for run-local heatmaps, not Figures.
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


type StageStatus = Literal["accepted", "rejected", "failed", "skipped"]


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


@dataclass(frozen=True)
class FrequencyResult:
    """One point's spectroscopy outcome and immutable relative native references.

    status is accepted, rejected (fit gates or stopped partial), or failed
    (Schedule or expected fitting failure). Partial reason is stopped.
    reason is None on acceptance, otherwise the diagnostic. frequency_mhz is
    this point's positive fitted MHz only on acceptance, never a prior seed.
    files lists every Completed widening attempt in acquisition order, including
    partial, rejected and failed fits; Schedule failures have no filename.
    References do not guarantee later existence. Unexpected writer/adapter/fit/
    plot exceptions propagate instead. probe_gains records actual spectroscopy
    gain for every acquisition attempt, including failed/partial attempts.
    scan_center_mhz is the physical-plus-IDW center before window offset.
    """

    status: StageStatus
    reason: str | None
    frequency_mhz: float | None
    files: tuple[str, ...]
    probe_gains: tuple[float, ...] = ()
    scan_center_mhz: float | None = None


@dataclass(frozen=True)
class RabiResult:
    """One point's Length-Rabi outcome, not a substitute for prior calibration.

    status is accepted/rejected/failed/skipped; reason is None only on acceptance.
    frequency_mhz is the accepted frequency used at this point (MHz), or None
    when frequency was not accepted and Rabi is skipped. It never carries a
    previous point's frequency. Accepted implies a current accepted frequency.
    pi_length_us/pi2_length_us are positive fitted durations only on acceptance;
    pi2 comes from the phase-aware fit, not pi/2. file is the relative native
    filename after Completed, including partial and failed/rejected fits,
    otherwise None. Partial returns rejected/stopped without fitting; expected
    fitting failures return failed/reason. Unexpected errors propagate.
    """

    status: StageStatus
    reason: str | None
    frequency_mhz: float | None
    pi_length_us: float | None
    pi2_length_us: float | None
    file: str | None


@dataclass(frozen=True)
class ReadoutResult:
    """Readout scan outcome and immutable relative native reference.

    status is accepted/rejected/failed/skipped (dependency or flux cadence).
    reason is None only on acceptance. frequency_mhz/gain are this point's
    accepted readout MHz/native gain only; nonaccepted results have no values.
    file names the Completed landscape, including rejection, otherwise None.
    Nonacceptance leaves the legal seed/prior readout carry unchanged.
    Partial returns rejected/stopped after saving, without SNR selection.
    Unexpected analysis/configuration/writer errors propagate.
    """

    status: StageStatus
    reason: str | None
    frequency_mhz: float | None
    gain: float | None
    file: str | None


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


def _accept_frequency(fit: FrequencyFit, tun: CalibrationTunables) -> bool:
    residual = fit.quality.normalized_residual_rms
    return (
        math.isfinite(fit.frequency_mhz)
        and fit.frequency_mhz > 0
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
) -> None:
    if 0 < fit.width_mhz <= float(np.ptp(result.freqs)) and trusted_curve(
        fit.observed, fit.fitted, 0.1
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
    carry.cur_m, carry.m_ratio = tracker.cur_m, tracker.m_ratio
    gain = tracker.probe_gain(float(inputs.qub_pulse.gain), inputs.tun)
    filenames: list[str] = []
    gains: list[float] = []
    status: StageStatus = "rejected"
    reason = "No accepted frequency fit"
    for attempt, widen in enumerate(inputs.tun.qf_widen, start=1):
        if scope.cancelled:
            return FrequencyResult(
                "skipped", "Point cancelled", None, tuple(filenames), tuple(gains), hint
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
        accepted = _accept_frequency(fit, inputs.tun)
        if accepted:
            carry.last = replace(prior, f01_mhz=fit.frequency_mhz)
            carry.f01_by_flux.append((flux, fit.frequency_mhz))
            tracker.finish_point(flux, fit.frequency_mhz, tun=inputs.tun)
            _observe_linewidth(tracker, fit, result, gain, inputs.point_index)
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
            )
        status, reason = "rejected", "No accepted frequency fit"
    if reason != "stopped":
        tracker.finish_point(flux, None, tun=inputs.tun)
    return FrequencyResult(status, reason, None, tuple(filenames), tuple(gains), hint)


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
    finite us; relax_delay remains five times it. inputs supplies captured
    modules/policy. carry retains readout and frequency; only fitted pi/pi2 change
    on acceptance. inputs.snapshot is the fixed historical entry/point state
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
    ro = inputs.readout.model_copy(deep=True)
    apply_readout_calibration(ro, prior)
    start = inputs.tun.rabi_start_factor * prior.pi_length_us
    stop = inputs.tun.rabi_stop_factor * prior.pi_length_us
    cfg = LenRabiCfg(
        reps=inputs.tun.rabi_reps,
        rounds=inputs.tun.rabi_rounds,
        relax_delay=5 * t1_estimate_us,
        qub_pulse=drive,
        readout=ro,
        length=SweepCfg(start=start, stop=stop, expts=51, step=(stop - start) / 50),
    )
    outcome = await scope.run(LenRabiExp().run, cfg, name="Rabi")
    if isinstance(outcome, Failed):
        return RabiResult("failed", outcome.reason, f01, None, None, None)
    # Save before fitting, so rejected fits retain their native trace.
    save_native(
        outcome,
        inputs.directory / "rabi.h5",
        spec=LenRabiExp.AXES_SPEC,
        snapshot=inputs.snapshot,
        identity=scope.identity,
    )
    if outcome.partial_result:
        return RabiResult("rejected", "stopped", f01, None, None, "rabi.h5")
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
        return RabiResult("failed", fit.reason, f01, None, None, "rabi.h5")
    residual = fit.quality.normalized_residual_rms
    accepted = (
        math.isfinite(fit.pi_length_us)
        and fit.pi_length_us > 0
        and math.isfinite(fit.error_us)
        and fit.error_us >= 0
        and fit.error_us / fit.pi_length_us < inputs.tun.max_rabi_rel_err
        and residual is not None
        and residual < inputs.tun.max_fit_residual
    )
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
        return RabiResult("rejected", "Rabi fit rejected", f01, None, None, "rabi.h5")
    # Update only accepted pulse durations and show the fitted oscillation.
    carry.last = CalibratedPi2(
        prior.f01_mhz,
        fit.pi_length_us,
        prior.readout_freq_mhz,
        prior.readout_gain,
        fit.pi2_length_us,
    )
    return RabiResult(
        "accepted", None, f01, fit.pi_length_us, fit.pi2_length_us, "rabi.h5"
    )


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

    Resolve the latest accepted pi pulse, retaining its paired MHz even when
    this point's Rabi fails. Without any accepted pulse, return a dependency
    skip. Caller uses readout_due to decide flux cadence. Accepted rabi updates
    the pulse and its producer point in carry before acquisition. Timing is
    positive finite us, with relax_delay five times it. carry's readout changes
    only on accepted finite SNR. Failure/rejection retains legal prior
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
    ro = inputs.readout.model_copy(deep=True)
    apply_readout_calibration(ro, prior)
    cfg = FreqGainCfg(
        reps=inputs.tun.ro_reps,
        rounds=inputs.tun.ro_rounds,
        relax_delay=5 * t1_estimate_us,
        qub_pulse=drive,
        readout=ro,
        freq=SweepCfg(
            start=prior.readout_freq_mhz - 5.0,
            stop=prior.readout_freq_mhz + 5.0,
            expts=51,
            step=10.0 / 50,
        ),
        gain=SweepCfg(start=0.0, stop=0.2, expts=31, step=0.2 / 30),
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
    draw_readout(
        plots,
        outcome.value.result.freqs,
        outcome.value.result.gains,
        outcome.value.result.signals,
        inputs_flux=(inputs.flux, inputs.flux_unit),
        directory=inputs.directory,
    )
    signals = outcome.value.result.signals
    accepted = np.isfinite(signals) & (signals >= inputs.tun.min_readout_snr)
    if not np.any(accepted):
        return ReadoutResult(
            "rejected", "No readout SNR meets threshold", None, None, "readout.h5"
        )
    row, column = np.unravel_index(
        np.argmax(np.where(accepted, signals, -np.inf)), signals.shape
    )
    frequency = float(outcome.value.result.freqs[row])
    gain = float(outcome.value.result.gains[column])
    carry.last = replace(prior, readout_freq_mhz=frequency, readout_gain=gain)
    carry.readout_source = CalibrationSource(
        source_index=inputs.point_index, source_flux=inputs.flux, fallback=False
    )
    return ReadoutResult("accepted", None, frequency, gain, "readout.h5")


def current_point_calibration(
    rabi: RabiResult,
    carry: CalCarry,
    *,
    point_index: int,
    flux: float,
    need: Literal["latest", "now"] = "latest",
) -> CalibratedPi2 | None:
    """Resolve pi/pi2 with latest or current-point-only dependency semantics.

    Accepted Rabi stores its paired drive frequency and durations with producer
    index/flux in carry. Rejected/skipped Rabi leaves that pulse intact; latest
    can reuse it without an age limit, while now returns None. A later frequency
    update never retunes an older pi module. Readout uses its independent latest
    accepted value or the fixed host seed. No seed pi/2 is fabricated.
    """
    prior = carry.seed if carry.last is None else carry.last
    if rabi.status == "accepted":
        if (
            rabi.frequency_mhz is None
            or rabi.pi_length_us is None
            or rabi.pi2_length_us is None
        ):
            raise ValueError("Accepted Rabi requires current frequency, pi and pi2")
        carry.last_pi = CalibratedPi2(
            rabi.frequency_mhz,
            rabi.pi_length_us,
            prior.readout_freq_mhz,
            prior.readout_gain,
            rabi.pi2_length_us,
        )
        carry.pi_source = CalibrationSource(
            source_index=point_index, source_flux=flux, fallback=False
        )
    if rabi.status != "accepted" and need == "now":
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
