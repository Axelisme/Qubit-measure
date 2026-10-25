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

from zcu_lab.v2.twotone.freq.core import (
    FreqCfg,
)
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
    draw_frequency,
    draw_rabi,
    draw_readout,
    retain_frequency_fit,
    retain_trace,
)
from zcu_lab.workflows.common.calibration_results import (
    FrequencyResult as FrequencyResult,
)
from zcu_lab.workflows.common.calibration_results import (
    FrequencyScan,
    StageStatus,
)
from zcu_lab.workflows.common.calibration_results import (
    RabiResult as RabiResult,
)
from zcu_lab.workflows.common.calibration_results import (
    ReadoutResult as ReadoutResult,
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
from zcu_lab.workflows.common.frequency_scan import (
    accept_frequency_fit,
    frequency_probe_gains,
    measure_frequency_scan,
    observe_frequency_linewidth,
)
from zcu_lab.workflows.common.frequency_tracking import (
    FrequencyTracker,
    trusted_curve,
)
from zcu_lab.workflows.common.rabi_feedback import RabiDriveFeedback
from zcu_lab.workflows.common.rabi_measurement import measure_planned_rabi
from zcu_lab.workflows.common.readout_analysis import accepted_readout_optimum
from zcu_lab.workflows.common.saving import save_native


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
    last_pi retains the frequency paired with accepted pi/pi2; pi_source and
    readout_source identify their producer points, or None before acceptance.
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

    qub_pulse is the Rabi/downstream pi drive; probe_pulse independently supplies
    spectroscopy waveform and initial gain. readout is shared pulse-backed readout.
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
    probe_pulse: PulseCfg
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
    span: float,
    offset: float = 0.0,
) -> FreqCfg:
    drive = inputs.probe_pulse.model_copy(deep=True)
    drive.set_param("gain", gain)
    ro = inputs.readout.model_copy(deep=True)
    apply_readout_calibration(ro, prior)
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


async def calibrate_frequency(
    scope: Scope,
    *,
    plots: Plots,
    inputs: CalibrationInputs,
    flux: float,
    carry: CalCarry,
) -> FrequencyResult:
    """Locate f01 with a coarse survey followed by a lower-power fine scan.

    two_step selects the candidate nearest the physical-plus-IDW center, isolates
    it from neighbouring peaks and recenters the fine scan on the coarse fit.
    Only coarse failures use qf_widen. A truncated coarse peak must be rescanned.
    Fine results alone enter acceptance, IDW, recovery history and linewidth
    feedback. single retains whole-window fitting and the original widening.
    Save every Completed attempt before analysis, including partial results.
    scans records each stage's actual gain/span/file and all candidates.
    Partial results do not enter fitting or feedback. Invalid flux raises before
    acquisition; unexpected acquisition, writer and analysis errors propagate.
    """
    if not math.isfinite(flux):
        raise ValueError("Calibration flux must be finite")
    prior = carry.seed if carry.last is None else carry.last
    tracker = carry.frequency_tracker
    if tracker is None:
        raise ValueError("Frequency calibration requires a captured physical predictor")
    hint = tracker.begin_point(flux, seed_flux=carry.seed_flux)
    predictor_model = tracker.model_snapshot()
    carry.cur_m, carry.m_ratio = tracker.cur_m, tracker.m_ratio
    tun = inputs.tun
    coarse_gain, fine_gain, adaptive_gain = frequency_probe_gains(inputs, tracker)
    first_stage: Literal["coarse", "fine", "single"] = (
        "coarse" if tun.frequency_mode == "two_step" else "single"
    )
    pending: list[tuple[Literal["coarse", "fine", "single"], float]] = [
        (first_stage, widen) for widen in tun.qf_widen
    ]
    center = hint
    scans: list[FrequencyScan] = []
    status: StageStatus = "rejected"
    reason = "No accepted frequency fit"
    while pending:
        stage, widen = pending.pop(0)
        if scope.cancelled:
            return FrequencyResult(
                "skipped",
                "Point cancelled",
                None,
                tuple(scan.file for scan in scans if scan.file is not None),
                tuple(scan.gain for scan in scans),
                hint,
                predictor_model,
                tracker.last_recovery,
                tuple(scans),
            )
        gain = (
            coarse_gain
            if stage == "coarse"
            else fine_gain
            if stage == "fine"
            else adaptive_gain
        )
        span = (
            (tun.fine_span_mhz or tun.span_mhz)
            if stage == "fine"
            else tun.span_mhz
            * widen
            * (tun.coarse_span_factor if stage == "coarse" else 1)
        )
        cfg = _frequency_cfg(
            inputs,
            prior,
            center,
            gain,
            span,
            0.0 if stage == "fine" else tun.detune_center_mhz,
        )
        filename = (
            f"frequency-{len(scans) + 1:02d}.h5"
            if stage == "single"
            else f"frequency-{stage}-{len(scans) + 1:02d}.h5"
        )
        scan, fit = await measure_frequency_scan(
            scope,
            cfg,
            FrequencyScan(
                stage,
                filename,
                gain,
                span,
                center + (0.0 if stage == "fine" else tun.detune_center_mhz),
            ),
            inputs,
        )
        scans.append(scan)
        if isinstance(fit, Failed):
            status = (
                "rejected"
                if fit.reason in ("stopped", "Selected peak is truncated")
                else "failed"
            )
            reason = fit.reason
            continue
        result = fit.source
        retain_frequency_fit(
            carry.frequency_traces, fit, inputs.point_index, flux, hint
        )
        draw_frequency(
            plots,
            carry.frequency_traces,
            carry.f01_by_flux,
            flux_unit=inputs.flux_unit,
            directory=inputs.directory,
        )
        if stage == "coarse":
            if not math.isfinite(fit.frequency_mhz) or not float(
                np.min(fit.freqs)
            ) <= fit.frequency_mhz <= float(np.max(fit.freqs)):
                status, reason = (
                    "rejected",
                    "Coarse fitted center is outside its candidate interval",
                )
                continue
            center = fit.frequency_mhz
            # Drop unused coarse widening rungs. A fine failure ends this point.
            pending = [("fine", 1.0)]
            continue
        if accept_frequency_fit(fit, result, tun):
            carry.last = replace(prior, f01_mhz=fit.frequency_mhz)
            carry.f01_by_flux.append((flux, fit.frequency_mhz))
            tracker.finish_point(flux, fit.frequency_mhz, tun=tun)
            observe_frequency_linewidth(
                tracker, fit, result, gain, inputs.point_index, tun
            )
            draw_frequency(
                plots,
                carry.frequency_traces,
                carry.f01_by_flux,
                flux_unit=inputs.flux_unit,
                directory=inputs.directory,
            )
            return FrequencyResult(
                "accepted",
                None,
                fit.frequency_mhz,
                tuple(scan.file for scan in scans if scan.file is not None),
                tuple(scan.gain for scan in scans),
                hint,
                predictor_model,
                tracker.last_recovery,
                tuple(scans),
            )
        status, reason = "rejected", "No accepted frequency fit"
    if reason != "stopped":
        tracker.finish_point(flux, None, tun=tun)
    return FrequencyResult(
        status,
        reason,
        None,
        tuple(scan.file for scan in scans if scan.file is not None),
        tuple(scan.gain for scan in scans),
        hint,
        predictor_model,
        tracker.last_recovery,
        tuple(scans),
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
                pi2_drive_gain=rabi.pi2_drive_gain,
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
