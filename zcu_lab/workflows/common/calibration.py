"""common calibration helpers for lab workflows."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel
from zcu_tools.experiment.metadata import RunSnapshot
from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.workflows import Failed
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import SweepCfg
from zcu_tools.program.v2.modules import PulseCfg, PulseReadoutCfg, ReadoutCfg
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
from zcu_lab.workflows.common.calibration_results import StageStatus as StageStatus
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
from zcu_lab.workflows.common.rabi_feedback import RabiDriveFeedback
from zcu_lab.workflows.common.rabi_measurement import measure_planned_rabi
from zcu_lab.workflows.common.readout_analysis import accepted_readout_optimum
from zcu_lab.workflows.common.saving import save_native
from zcu_lab.workflows.common.summary import SummaryAccumulator, observe_operation


@dataclass
class CalCarry:
    """Coroutine-local calibration carry retained between committed points.

    seed is the legal host scalar seed; last contains the latest accepted
    scalar parts or None initially. Frequency, Rabi and readout update only
    their own fields. f01_by_flux contains accepted (device setpoint, MHz)
    frequency history even when a later stage fails. frequency_traces/rabi_traces
    retain one projected sweep row per point for run-local heatmaps, not Figures.
    amp_traces retains gain-scan rows; last_amp_gain is the last accepted Amp pi
    gain and centers the next amp-only window, or None before Amp acceptance.
    rabi_drive holds the pi-product history and query-aged gain controller.
    readout_source identifies the accepted readout producer, or None before
    acceptance. Downstream pulse consumers resolve only this point's Rabi.
    Widening replaces the current point's frequency row. Pause preserves this carry;
    only explicit point commits persist reports. There are no stage checkpoints.
    frequency_tracker owns the run-local physical overlay, IDW and linewidth
    feedback. seed_flux is the seed pulse's calibration setpoint in the plan's
    flux unit; None disables matrix-element feed-forward. cur_m is |<0|n|1>|,
    and m_ratio is relative to that seed setpoint, never the first scan point.
    Both are None before prediction; ratio is also None for a zero reference.
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

    qub_pulse is the Rabi/downstream pi drive; probe_pulse independently supplies
    spectroscopy waveform and initial gain. readout is shared pulse-backed readout.
    tun is this invocation's ordinary generation/acceptance/cadence policy.
    The wrapper is frozen; nested modules are copied before applying scalars.
    snapshot is the fixed historical entry/point/role state used by all three
    native saves, not a live source. directory is the allocated commit directory.
    Coherence and MIST reuse fixed modules/snapshot/directory across stages,
    replacing tun from tun.current.cal before each stage. summary optionally
    collects delivered operation facts for the normal-return report. A stage does not
    recapture sources or refresh its own policy during acquisition.
    flux/flux_unit label the current point's history rows in device coordinates.
    """

    qub_pulse: PulseCfg
    probe_pulse: PulseCfg
    readout: PulseReadoutCfg
    tun: CalibrationTunables
    snapshot: RunSnapshot
    directory: Path
    point_index: int
    flux: float
    flux_unit: str
    summary: SummaryAccumulator | None = None


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


async def calibrate_rabi(
    scope: Scope,
    *,
    plots: Plots,
    inputs: CalibrationInputs,
    frequency: FrequencyResult,
    carry: CalCarry,
    t1_estimate_us: float,
) -> RabiResult:
    """Calibrate current pi/pi2 with the selected length/gain Rabi mode.

    len_only retains length fitting. len_then_amp requires accepted LenRabi,
    then uses its length for gain calibration. amp_only skips length acquisition
    and fixes length to the resolved target, including the seed/window default. Both Amp modes
    require accepted current frequency; failures never borrow a previous pulse.
    Partial results save raw but do not proceed to another acquisition.
    """
    if inputs.tun.rabi_mode != "amp_only":
        rabi = await calibrate_len_rabi(
            scope,
            plots=plots,
            inputs=inputs,
            frequency=frequency,
            carry=carry,
            t1_estimate_us=t1_estimate_us,
        )
        if inputs.tun.rabi_mode == "len_only" or rabi.status != "accepted":
            return rabi
    else:
        _validate_calibration_inputs(inputs.flux, t1_estimate_us)
        if frequency.status != "accepted":
            return RabiResult(
                "skipped",
                "Requires this point's accepted frequency",
                None,
                None,
                None,
                None,
            )
        length = inputs.tun.resolve_target_pi_length_us(carry.seed.pi_length_us)
        rabi = RabiResult(
            "accepted",
            None,
            frequency.frequency_mhz,
            length,
            None,
            None,
            target_pi_length_us=length,
        )
    if scope.cancelled:
        return replace(
            rabi,
            status="skipped",
            reason="Point cancelled",
            pi_length_us=None,
            pi2_length_us=None,
        )
    return await calibrate_amp_rabi(
        scope,
        plots=plots,
        inputs=inputs,
        carry=carry,
        rabi=rabi,
        t1_estimate_us=t1_estimate_us,
    )


async def calibrate_len_rabi(
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
    target_pi_us = inputs.tun.resolve_target_pi_length_us(carry.seed.pi_length_us)
    if inputs.tun.rabi_drive_gain_mode == "auto_pi_product":
        gain = carry.rabi_drive.select_gain(
            carry.seed.pi_length_us * float(inputs.qub_pulse.gain),
            target_pi_us,
            m_ratio=carry.m_ratio,
        )
        drive.set_param("gain", gain)
    ro = inputs.readout.model_copy(deep=True)
    apply_readout_calibration(ro, prior)
    start = max(
        inputs.tun.rabi_min_start_us, inputs.tun.rabi_start_factor * prior.pi_length_us
    )
    stop = max(
        inputs.tun.rabi_min_stop_us,
        inputs.tun.rabi_stop_factor * prior.pi_length_us,
        inputs.tun.rabi_stop_factor * target_pi_us,
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
    outcome = await scope.run(measure_planned_rabi, cfg, name="Rabi")
    observe_operation(inputs.summary, outcome)
    diagnostics = RabiResult(
        "failed",
        None,
        f01,
        None,
        None,
        None,
        gain,
        target_pi_length_us=target_pi_us,
        requested_sweep=cfg.length,
        planned_sweep=(
            None
            if isinstance(outcome, Failed) or outcome.value.cfg is None
            else outcome.value.cfg.length
        ),
    )
    if isinstance(outcome, Failed):
        return replace(diagnostics, reason=outcome.reason)
    # Save before fitting, so rejected fits retain their native trace.
    save_native(
        outcome,
        inputs.directory / "rabi.h5",
        spec=LenRabiExp.AXES_SPEC,
        snapshot=inputs.snapshot,
        identity=scope.identity,
    )
    if outcome.partial_result:
        return replace(diagnostics, status="rejected", reason="stopped", file="rabi.h5")
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
        return replace(diagnostics, reason=fit.reason, file="rabi.h5")
    residual = fit.quality.normalized_residual_rms
    quality_trusted = (
        math.isfinite(fit.pi_length_us)
        and fit.pi_length_us > 0
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
    too_slow = quality_trusted and fit.pi_length_us > 0.9 * float(
        np.max(result.lengths)
    )
    accepted = quality_trusted and not too_fast and not too_slow
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
        if inputs.tun.rabi_drive_gain_mode == "auto_pi_product":
            if too_fast:
                carry.rabi_drive.observe_too_fast(
                    fit.pi_length_us, gain, max(target_pi_us, 0.03)
                )
            elif too_slow:
                carry.rabi_drive.observe_too_slow(
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
        return replace(
            diagnostics,
            status="rejected",
            reason=reason,
            file="rabi.h5",
            too_fast_pi_length_us=fit.pi_length_us if too_fast else None,
            too_slow_pi_length_us=fit.pi_length_us if too_slow else None,
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
    return replace(
        diagnostics,
        status="accepted",
        pi_length_us=fit.pi_length_us,
        pi2_length_us=pi2,
        file="rabi.h5",
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
    *,
    plots: Plots,
    inputs: CalibrationInputs,
    rabi: RabiResult,
    carry: CalCarry,
    t1_estimate_us: float,
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
    _validate_calibration_inputs(0.0, t1_estimate_us)
    cal = current_pi_calibration(rabi, carry)
    if cal is None:
        return ReadoutResult(
            "skipped", "Requires this point's accepted pi pulse (NOW)", None, None, None
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
    observe_operation(inputs.summary, outcome)
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
