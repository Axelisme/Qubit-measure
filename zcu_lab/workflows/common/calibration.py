"""common calibration helpers for lab workflows."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
from pydantic import BaseModel
from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.workflows import Failed
from zcu_tools.program.v2 import SweepCfg
from zcu_tools.program.v2.modules import ReadoutCfg
from zcu_tools.utils.process import SmoothingOptions, smooth_signal_nd

from zcu_lab.v2.twotone.rabi.len_rabi.core import (
    LenRabiCfg,
    LenRabiExp,
)
from zcu_lab.v2.twotone.ro_optimize.freq_gain.core import (
    FreqGainCfg,
    FreqGainExp,
)
from zcu_lab.workflows.common.amp_calibration import calibrate_amp_rabi
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
    draw_readout,
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
from zcu_lab.workflows.common.calibration_results import StageStatus as StageStatus
from zcu_lab.workflows.common.calibration_stage_types import (
    CalibrationPoint,
    CalibrationPulses,
    RabiGuess,
    RabiStageResult,
    ReadoutPulses,
)
from zcu_lab.workflows.common.calibration_tunables import (
    CalibrationTunables as CalibrationTunables,
)
from zcu_lab.workflows.common.calibration_types import (
    Calibrated as Calibrated,
)
from zcu_lab.workflows.common.calibration_types import (
    CalibratedPi2 as CalibratedPi2,
)
from zcu_lab.workflows.common.frequency_calibration import (
    calibrate_frequency as calibrate_frequency,
)
from zcu_lab.workflows.common.frequency_tracking import (
    FrequencyTracker,
    trusted_curve,
)
from zcu_lab.workflows.common.gain import GainClamp, clamp_gain
from zcu_lab.workflows.common.rabi_feedback import RabiDriveFeedback
from zcu_lab.workflows.common.rabi_measurement import measure_planned_rabi
from zcu_lab.workflows.common.readout_analysis import accepted_readout_optimum
from zcu_lab.workflows.common.saving import save_native
from zcu_lab.workflows.common.stages import StageEnv
from zcu_lab.workflows.common.summary import observe_operation


@dataclass
class CalCarry:
    """Run-local calibration history and feedback controllers across points.

    seed is the legal host scalar seed; last holds the latest accepted scalar
    parts, or None initially. f01_by_flux holds accepted (device setpoint, MHz)
    frequency history. frequency_traces/rabi_traces/amp_traces hold one
    projected sweep row per point for run-local heatmaps, not Figures.
    last_amp_gain is the last accepted Amp pi gain, which centres the next
    amp-only window, or None before Amp acceptance. readout_source names the
    accepted readout producer, or None before acceptance.
    frequency_tracker and rabi_drive are feedback controllers handed to the
    stages that observe them during a measurement. seed_flux is the seed pulse's
    calibration setpoint in the plan's flux unit; None disables matrix-element
    feed-forward. Only calibrate_point updates the history fields, from stage
    results. Pause preserves this carry; there are no stage checkpoints.
    """

    seed: Calibrated
    seed_flux: float | None = field(default=None, kw_only=True)
    last: Calibrated | None = None
    f01_by_flux: list[tuple[float, float]] = field(default_factory=list)
    frequency_traces: list[CalibrationTrace] = field(default_factory=list)
    rabi_traces: list[CalibrationTrace] = field(default_factory=list)
    amp_traces: list[CalibrationTrace] = field(default_factory=list)
    last_amp_gain: float | None = None
    rabi_drive: RabiDriveFeedback = field(default_factory=RabiDriveFeedback)
    readout_source: CalibrationSource | None = None
    frequency_tracker: FrequencyTracker | None = None


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


def apply_readout_calibration(
    readout: ReadoutCfg,
    cal: Calibrated,
    gain_clamped: list[GainClamp] | None = None,
) -> None:
    """Apply accepted readout scalars to a caller-owned native cfg in place.

    Updates the shared generator/ADC frequency in MHz and native readout gain.
    Clip gain to [0, 1], appending any before/after change to gain_clamped.
    Callers must copy any template they need to retain before calling.
    Native set_param errors propagate. No acquisition or saving occurs.
    """
    readout.set_param("freq", cal.readout_freq_mhz)
    readout.set_param(
        "gain", clamp_gain(cal.readout_gain, "readout", gain_clamped, lower=0.0)
    )


def _validate_calibration_inputs(flux: float, t1_estimate_us: float) -> None:
    """Reject nonfinite flux and nonpositive/nonfinite T1 estimates in µs."""
    if not math.isfinite(t1_estimate_us) or t1_estimate_us <= 0:
        raise ValueError("T1 estimate must be positive and finite")
    if not math.isfinite(flux):
        raise ValueError("Calibration flux must be finite")


async def calibrate_rabi(
    scope: Scope,
    env: StageEnv,
    pulses: CalibrationPulses,
    settings: CalibrationTunables,
    *,
    guess: RabiGuess,
    rabi_drive: RabiDriveFeedback,
) -> RabiStageResult:
    """Calibrate this point's pi/pi2, retaining Len acceptance if Amp rejects.

    pulses supplies prior length/template gain and calibrated readout, settings
    selects len_only/len_then_amp/amp_only. guess supplies current frequency and
    timing/gain predictions. Only the named feedback controller is mutated;
    history and point records belong to the caller. Partial saves raw and stops
    before Amp; unexpected acquisition, saving and analysis errors propagate.
    """
    if settings.rabi_mode != "amp_only":
        result = await calibrate_len_rabi(
            scope, env, pulses, settings, guess=guess, rabi_drive=rabi_drive
        )
        result = replace(result, length_record=result.record)
        rabi = result.record
        if settings.rabi_mode == "len_only" or rabi.status != "accepted":
            return result
    else:
        _validate_calibration_inputs(guess.point.flux, guess.relax_us)
        if guess.frequency_mhz is None:
            return RabiStageResult(
                record=RabiResult(
                    status="skipped",
                    reason="Requires this point's accepted frequency",
                    frequency_mhz=None,
                    pi_length_us=None,
                    pi2_length_us=None,
                    file=None,
                )
            )
        rabi = RabiResult(
            status="accepted",
            reason=None,
            frequency_mhz=guess.frequency_mhz,
            pi_length_us=guess.target_pi_us,
            pi2_length_us=None,
            file=None,
            target_pi_length_us=guess.target_pi_us,
        )
        result = RabiStageResult(record=rabi)
    if scope.cancelled:
        return replace(
            result,
            record=replace(
                rabi,
                status="skipped",
                reason="Point cancelled",
                pi_length_us=None,
                pi2_length_us=None,
            ),
        )
    amp_result = await calibrate_amp_rabi(
        scope, env, pulses, settings, guess=guess, rabi=rabi
    )
    return replace(
        amp_result, length_record=result.length_record, len_trace=result.len_trace
    )


async def calibrate_len_rabi(
    scope: Scope,
    env: StageEnv,
    pulses: CalibrationPulses,
    settings: CalibrationTunables,
    *,
    guess: RabiGuess,
    rabi_drive: RabiDriveFeedback,
) -> RabiStageResult:
    """Measure Rabi oscillations to calibrate pi and pi/2 pulse durations here.

    frequency must be the current point's result, never a prior frequency.
    Missing acceptance returns a named skipped result. t1_estimate_us is positive
    finite historical smoothed us. Relaxation uses the Rabi auto/fixed knobs.
    inputs supplies captured modules/policy. Auto gain blends pi-product prediction
    with confidence-aged log feedback; fixed mode retains template gain. Sweep
    endpoints cover prior and target pi durations, then use a clock-planned grid.
    Only accepted fits update the usable pi product. Quality-trusted fits below
    .03 us or above 90% of measured maximum remain rejected but propose lower
    or higher auto gain, without changing the usable pulse. inputs.snapshot is the fixed historical entry/point state
    used to save rabi.h5 after Completed, including rejected fits. point_index
    must be a nonnegative integer (not bool), the zero-based flux index; invalid
    indexes raise ValueError at entry. The title shows its flux setpoint. Lengths
    stay in microseconds and the projected signal has arbitrary units.
    Partial saves then returns rejected/stopped without fitting or carry updates.
    Schedule and expected fit failures are named failed; unexpected errors
    propagate. Invalid timing raises before any acquisition. Await in a workflow.
    """
    # Require a valid plot index and this point's accepted frequency.
    if type(guess.point.index) is not int or guess.point.index < 0:
        raise ValueError("Rabi point index must be a nonnegative integer")
    _validate_calibration_inputs(0.0, guess.relax_us)
    if guess.frequency_mhz is None:
        return RabiStageResult(
            record=RabiResult(
                status="skipped",
                reason="Requires this point's accepted frequency",
                frequency_mhz=None,
                pi_length_us=None,
                pi2_length_us=None,
                file=None,
            )
        )
    f01 = guess.frequency_mhz
    # Sweep duration at the accepted frequency, retaining legal readout.
    drive = pulses.qub.model_copy(deep=True)
    drive.set_param("freq", f01)
    gain_clamped: list[GainClamp] = []
    target_pi_us = guess.target_pi_us
    gain = (
        rabi_drive.select_gain(
            guess.seed_product,
            target_pi_us,
            m_ratio=guess.m_ratio,
            gain_clamped=gain_clamped,
        )
        if settings.rabi_drive_gain_mode == "auto_pi_product"
        else float(drive.gain)
    )
    gain = clamp_gain(gain, "rabi.drive", gain_clamped)
    drive.set_param("gain", gain)
    ro = pulses.readout.model_copy(deep=True)
    ro.set_param("gain", clamp_gain(float(ro.gain), "readout", gain_clamped, lower=0.0))
    start = max(
        settings.rabi_min_start_us, settings.rabi_start_factor * guess.prior_pi_us
    )
    stop = max(
        settings.rabi_min_stop_us,
        settings.rabi_stop_factor * guess.prior_pi_us,
        settings.rabi_stop_factor * target_pi_us,
    )
    cfg = LenRabiCfg(
        reps=settings.rabi_reps,
        rounds=settings.rabi_rounds,
        acquire_retry=settings.rabi_acquire_retry,
        earlystop_snr=settings.rabi_earlystop_snr,
        relax_delay=(
            max(
                settings.rabi_relax_min_us,
                settings.rabi_relax_factor * guess.relax_us,
            )
            if settings.rabi_relax_mode == "auto"
            else settings.rabi_relax_delay_us
        ),
        qub_pulse=drive,
        readout=ro,
        length=SweepCfg(
            start=start,
            stop=stop,
            expts=settings.rabi_points,
            step=(stop - start) / (settings.rabi_points - 1),
        ),
    )
    outcome = await scope.run(measure_planned_rabi, cfg, name="Rabi")
    observe_operation(env.summary, outcome)
    diagnostics = RabiResult(
        "failed",
        None,
        f01,
        None,
        None,
        None,
        gain,
        target_pi_length_us=target_pi_us,
        gain_clamped=tuple(gain_clamped),
        requested_sweep=cfg.length,
        planned_sweep=(
            None
            if isinstance(outcome, Failed) or outcome.value.cfg is None
            else outcome.value.cfg.length
        ),
    )
    if isinstance(outcome, Failed):
        return RabiStageResult(record=replace(diagnostics, reason=outcome.reason))
    # Save before fitting, so rejected fits retain their native trace.
    save_native(
        outcome,
        scope.dir / "rabi.h5",
        spec=LenRabiExp.AXES_SPEC,
        snapshot=env.snapshot,
        identity=scope.identity,
    )
    if outcome.partial_result:
        return RabiStageResult(
            record=replace(
                diagnostics, status="rejected", reason="stopped", file="rabi.h5"
            )
        )
    result = outcome.value.result
    fit = analyze_len_rabi(result)
    trace = CalibrationTrace(
        guess.point.index,
        guess.point.flux,
        result.lengths,
        result.signals.real if isinstance(fit, Failed) else fit.observed,
    )
    if isinstance(fit, Failed):
        return RabiStageResult(
            record=replace(diagnostics, reason=fit.reason, file="rabi.h5"),
            len_trace=trace,
        )
    residual = fit.quality.normalized_residual_rms
    quality_trusted = (
        math.isfinite(fit.pi_length_us)
        and fit.pi_length_us > 0
        and trusted_curve(
            fit.observed, fit.fitted, settings.rabi_max_mean_residual_ratio
        )
        and math.isfinite(fit.error_us)
        and fit.error_us >= 0
        and fit.error_us / fit.pi_length_us < settings.max_rabi_rel_err
        and residual is not None
        and residual < settings.max_fit_residual
    )
    too_fast = quality_trusted and fit.pi_length_us < 0.03
    too_slow = quality_trusted and fit.pi_length_us > 0.9 * float(
        np.max(result.lengths)
    )
    accepted = quality_trusted and not too_fast and not too_slow
    trace = replace(
        trace,
        observed=fit.observed,
        fitted=fit.fitted,
        accepted=fit.pi_length_us if accepted else None,
    )
    if not accepted:
        if settings.rabi_drive_gain_mode == "auto_pi_product":
            if too_fast:
                rabi_drive.observe_too_fast(
                    fit.pi_length_us, gain, max(target_pi_us, 0.03)
                )
            elif too_slow:
                rabi_drive.observe_too_slow(
                    fit.pi_length_us,
                    gain,
                    min(target_pi_us, 0.9 * float(np.max(result.lengths))),
                )
        reason = (
            "π shorter than minimum trusted length"
            if too_fast
            else "π exceeds 90% of measured sweep maximum"
            if too_slow
            else "Rabi fit rejected"
        )
        return RabiStageResult(
            record=replace(
                diagnostics,
                status="rejected",
                reason=reason,
                file="rabi.h5",
                too_fast_pi_length_us=fit.pi_length_us if too_fast else None,
                too_slow_pi_length_us=fit.pi_length_us if too_slow else None,
            ),
            len_trace=trace,
        )
    rabi_drive.observe(
        guess.point.index,
        fit.pi_length_us,
        gain,
        target_pi_us,
        control=settings.rabi_drive_gain_mode == "auto_pi_product",
    )
    pi2 = (
        fit.pi2_length_us
        if math.isfinite(fit.pi2_length_us) and fit.pi2_length_us >= 0.03
        else None
    )
    return RabiStageResult(
        record=replace(
            diagnostics,
            status="accepted",
            pi_length_us=fit.pi_length_us,
            pi2_length_us=pi2,
            file="rabi.h5",
        ),
        len_trace=trace,
    )


def readout_due(point_index: int, tun: CalibrationTunables) -> bool:
    """Return whether this zero-based flux point scans readout.

    The rule is point_index % tun.ro_every == 0; index zero naturally scans.
    Failed/rejected prior scans do not change the cadence or cause extra scans.
    The caller retains legal seed/prior readout on cadence skips and checks this
    point's current-pi dependency separately. Negative indexes raise ValueError.
    """
    if point_index < 0:
        raise ValueError("Readout point index must be nonnegative")
    return point_index % tun.ro_every == 0


async def calibrate_readout(
    scope: Scope,
    env: StageEnv,
    pulses: ReadoutPulses,
    settings: CalibrationTunables,
    *,
    point: CalibrationPoint,
    guess_us: float,
) -> ReadoutResult:
    """Scan readout frequency and gain to maximize discrimination after Rabi.

    Require this point's accepted pi pulse with its source frequency and gain.
    Failed Rabi skips this scan. Caller uses readout_due to decide cadence;
    a dependency skip does not postpone it. Relaxation follows smoothed T1 or
    the fixed readout knob.
    Search windows follow retained readout with configurable half widths and gain
    clipped to [0, 1]. A wavelet-smoothed map needs a unique interior prominent
    maximum meeting the SNR floor. Failure/rejection retains legal prior
    readout, without returning fabricated accepted values. inputs.snapshot is
    the fixed historical entry/point state used to save readout.h5 after Completed,
    including rejection. Partial saves then returns rejected/stopped without
    selecting a best point or updating carry. Invalid input/unexpected errors raise.
    Await in a workflow; cancellation remains Engine-owned.
    """
    _validate_calibration_inputs(0.0, guess_us)
    # Sweep readout gain and frequency with the calibrated pi drive.
    drive = pulses.pi.model_copy(deep=True)
    gain_clamped: list[GainClamp] = []
    drive.set_param("gain", clamp_gain(float(drive.gain), "pi", gain_clamped))
    ro = pulses.readout.model_copy(deep=True)
    ro.set_param("gain", clamp_gain(float(ro.gain), "readout", gain_clamped, lower=0.0))
    gain_start = clamp_gain(
        float(pulses.readout.gain) - settings.ro_gain_half_width,
        "readout.start",
        gain_clamped,
        lower=0.0,
    )
    gain_stop = clamp_gain(
        float(pulses.readout.gain) + settings.ro_gain_half_width,
        "readout.stop",
        gain_clamped,
        lower=0.0,
    )
    cfg = FreqGainCfg(
        reps=settings.ro_reps,
        rounds=settings.ro_rounds,
        acquire_retry=settings.ro_acquire_retry,
        relax_delay=(
            max(settings.ro_relax_min_us, settings.ro_relax_factor * guess_us)
            if settings.ro_relax_mode == "auto"
            else settings.ro_relax_delay_us
        ),
        skew_penalty=settings.ro_skew_penalty,
        qub_pulse=drive,
        readout=ro,
        freq=SweepCfg(
            start=float(pulses.readout.freq) - settings.ro_freq_half_width_mhz,
            stop=float(pulses.readout.freq) + settings.ro_freq_half_width_mhz,
            expts=51,
            step=2 * settings.ro_freq_half_width_mhz / 50,
        ),
        gain=SweepCfg(
            start=gain_start,
            stop=gain_stop,
            expts=31,
            step=(gain_stop - gain_start) / 30,
        ),
    )
    outcome = await scope.run(FreqGainExp().run, cfg, name="readout")
    observe_operation(env.summary, outcome)
    if isinstance(outcome, Failed):
        return ReadoutResult(
            "failed", outcome.reason, None, None, None, tuple(gain_clamped)
        )
    # Save the landscape and choose its best point meeting the SNR threshold.
    save_native(
        outcome,
        scope.dir / "readout.h5",
        spec=FreqGainExp.AXES_SPEC,
        snapshot=env.snapshot,
        identity=scope.identity,
    )
    if outcome.partial_result:
        return ReadoutResult(
            "rejected", "stopped", None, None, "readout.h5", tuple(gain_clamped)
        )
    signals = np.abs(
        smooth_signal_nd(
            outcome.value.result.signals,
            options=SmoothingOptions(method="wavelet", sigma=1.0),
        )
    )
    draw_readout(
        env.plots,
        outcome.value.result.freqs,
        outcome.value.result.gains,
        signals,
        inputs_flux=(point.flux, point.flux_unit),
        directory=scope.dir,
    )
    optimum = accepted_readout_optimum(
        signals, outcome.value.result.freqs, outcome.value.result.gains
    )
    if (
        optimum is None
        or signals[optimum.freq_index, optimum.gain_index] < settings.min_readout_snr
    ):
        return ReadoutResult(
            "rejected",
            "Readout landscape has no trusted interior SNR optimum",
            None,
            None,
            "readout.h5",
            tuple(gain_clamped),
        )
    frequency = optimum.freq
    gain = clamp_gain(optimum.gain, "readout.optimum", gain_clamped, lower=0.0)
    return ReadoutResult(
        "accepted", None, frequency, gain, "readout.h5", tuple(gain_clamped)
    )


def current_pi_calibration(
    rabi: RabiResult | None, carry: CalCarry
) -> Calibrated | None:
    """Return this point's accepted pi with its own frequency/gain, without pi2.

    Nonaccepted or absent Rabi returns None; no prior pi is borrowed. An accepted
    record missing its frequency or pi duration raises ValueError. Readout uses
    its independent latest value or host seed. No pulse calibration is retained
    for use at a later point.
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
    rabi: RabiResult | None, carry: CalCarry
) -> CalibratedPi2 | None:
    """Return this point's accepted pi/pi2 pair, or None without a trusted pi2.

    Frequency, duration and gains come only from current accepted Rabi.
    Readout uses its independent accepted value or host seed. No prior pulse
    pair or fabricated pi/2 can satisfy this dependency.
    """
    cal = current_pi_calibration(rabi, carry)
    if cal is None or rabi is None or rabi.pi2_length_us is None:
        return None
    return CalibratedPi2(
        cal.f01_mhz,
        cal.pi_length_us,
        cal.readout_freq_mhz,
        cal.readout_gain,
        rabi.pi2_length_us,
        drive_gain=cal.drive_gain,
        pi2_drive_gain=rabi.pi2_drive_gain,
    )


def calibration_source(
    source: CalibrationSource | None, point_index: int
) -> CalibrationSource:
    """Mark whether a dependency came from this point, a prior point or host seed."""
    source = CalibrationSource() if source is None else source
    return source.model_copy(update={"fallback": source.source_index != point_index})
