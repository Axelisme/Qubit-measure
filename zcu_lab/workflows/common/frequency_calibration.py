"""Frequency calibration: coarse survey, candidate confirmation and feedback."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING, Literal

from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.workflows import Failed
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import SweepCfg

from zcu_lab.v2.twotone.freq.core import FreqCfg
from zcu_lab.workflows.common.calibration_analysis import FrequencyFit
from zcu_lab.workflows.common.calibration_plots import (
    draw_frequency,
    retain_frequency_fit,
)
from zcu_lab.workflows.common.calibration_results import (
    FrequencyCandidateAttempt,
    FrequencyResult,
    FrequencyScan,
    StageStatus,
)
from zcu_lab.workflows.common.frequency_scan import (
    accept_frequency_fit,
    acquire_frequency_scan,
    confirm_frequency_candidate,
    fit_coarse_candidates,
    fit_frequency_scan,
    frequency_probe_gains,
    observe_frequency_linewidth,
)
from zcu_lab.workflows.common.gain import GainClamp, clamp_gain

if TYPE_CHECKING:
    from zcu_tools.simulate.fluxonium.physical_fit import FluxoniumModelSnapshot

    from zcu_lab.workflows.common.calibration import (
        CalCarry,
        Calibrated,
        CalibrationInputs,
    )
    from zcu_lab.workflows.common.frequency_tracking import FrequencyTracker


def _frequency_cfg(
    inputs: CalibrationInputs,
    prior: Calibrated,
    center: float,
    gain: float,
    span: float,
    *,
    gain_clamped: list[GainClamp] | None = None,
) -> FreqCfg:
    from zcu_lab.workflows.common.calibration import apply_readout_calibration

    drive = inputs.probe_pulse.model_copy(deep=True)
    drive.set_param("gain", clamp_gain(gain, "probe", gain_clamped))
    ro = inputs.readout.model_copy(deep=True)
    apply_readout_calibration(ro, prior, gain_clamped)
    return FreqCfg(
        reps=inputs.tun.qf_reps,
        rounds=inputs.tun.qf_rounds,
        acquire_retry=inputs.tun.qf_acquire_retry,
        earlystop_snr=inputs.tun.qf_earlystop_snr,
        relax_delay=1.0,
        qub_pulse=drive,
        readout=ro,
        freq=SweepCfg(
            start=center - span / 2,
            stop=center + span / 2,
            expts=inputs.tun.qf_points,
            step=span / (inputs.tun.qf_points - 1),
        ),
    )


def _capture_frequency_prediction(
    carry: CalCarry, flux: float
) -> tuple[FrequencyTracker, float, FluxoniumModelSnapshot]:
    """Capture the point's predictor/model and update its matrix-element feedback."""
    if not math.isfinite(flux):
        raise ValueError("Calibration flux must be finite")
    tracker = carry.frequency_tracker
    if tracker is None:
        raise ValueError("Frequency calibration requires a captured physical predictor")
    hint = tracker.begin_point(flux, seed_flux=carry.seed_flux)
    predictor_model = tracker.model_snapshot()
    carry.cur_m, carry.m_ratio = tracker.cur_m, tracker.m_ratio
    return tracker, hint, predictor_model


def _plan_frequency_scan(
    inputs: CalibrationInputs,
    prior: Calibrated,
    stage_and_widen: tuple[Literal["coarse", "fine", "single"], float],
    center: float,
    gains: tuple[float, float, float],
    index: int,
) -> tuple[FreqCfg, FrequencyScan]:
    """Translate one scheduled stage into its cfg and native scan metadata."""
    stage, widen = stage_and_widen
    coarse_gain, fine_gain, adaptive_gain = gains
    gain = (
        coarse_gain
        if stage == "coarse"
        else fine_gain
        if stage == "fine"
        else adaptive_gain
    )
    tun = inputs.tun
    span = (
        (tun.fine_span_mhz or tun.span_mhz)
        if stage == "fine"
        else tun.span_mhz * widen * (tun.coarse_span_factor if stage == "coarse" else 1)
    )
    offset = 0.0 if stage == "fine" else tun.detune_center_mhz
    filename = (
        f"frequency-{index:02d}.h5"
        if stage == "single"
        else f"frequency-{stage}-{index:02d}.h5"
    )
    gain_clamped: list[GainClamp] = []
    cfg = _frequency_cfg(
        inputs, prior, center + offset, gain, span, gain_clamped=gain_clamped
    )
    return (
        cfg,
        FrequencyScan(
            stage,
            filename,
            float(cfg.qub_pulse.gain),
            span,
            center + offset,
            gain_clamped=tuple(gain_clamped),
        ),
    )


def _record_frequency_trace(
    fit: FrequencyFit | Failed,
    inputs: CalibrationInputs,
    carry: CalCarry,
    plots: Plots,
    flux_and_hint: tuple[float, float],
) -> None:
    """Plot a successful fit even when candidate confirmation or quality rejects it."""
    if isinstance(fit, Failed):
        return
    flux, hint = flux_and_hint
    retain_frequency_fit(carry.frequency_traces, fit, inputs.point_index, flux, hint)
    draw_frequency(
        plots,
        carry.frequency_traces,
        carry.f01_by_flux,
        flux_unit=inputs.flux_unit,
        directory=inputs.directory,
    )


def _accept_frequency_result(
    fit: FrequencyFit,
    inputs: CalibrationInputs,
    carry: CalCarry,
    plots: Plots,
    flux_and_hint: tuple[float, float],
    gain: float,
) -> bool:
    """Apply accepted-frequency feedback and redraw the accepted history."""
    if not accept_frequency_fit(fit, fit.source, inputs.tun):
        return False
    flux, _ = flux_and_hint
    tracker = carry.frequency_tracker
    if tracker is None:
        raise ValueError("Frequency feedback requires a physical predictor")
    prior = carry.seed if carry.last is None else carry.last
    carry.last = replace(prior, f01_mhz=fit.frequency_mhz)
    carry.f01_by_flux.append((flux, fit.frequency_mhz))
    tracker.finish_point(flux, fit.frequency_mhz, tun=inputs.tun)
    observe_frequency_linewidth(
        tracker, fit, fit.source, gain, inputs.point_index, inputs.tun
    )
    _record_frequency_trace(fit, inputs, carry, plots, flux_and_hint)
    return True


def _frequency_failure_status(
    fit: FrequencyFit | Failed, reason: str | None
) -> StageStatus:
    """Separate expected confirmation/gate rejection from acquisition/fit failure."""
    if reason in ("stopped", "Selected peak is truncated") or (
        reason is not None and reason.startswith("False coarse peak:")
    ):
        return "rejected"
    return "failed" if isinstance(fit, Failed) else "rejected"


async def calibrate_frequency(
    scope: Scope,
    *,
    plots: Plots,
    inputs: CalibrationInputs,
    flux: float,
    carry: CalCarry,
) -> FrequencyResult:
    """Survey and confirm at most max_candidates coarse targets at lower power.

    Try the predictor-nearest candidate first, then descending prominence.
    Reuse the survey for local fits; each valid target gets one fine scan.
    Reject absent fine peaks or a strongest peak farther than the coarse
    candidate's observable width from its fitted center. A failed fit or gate
    tries the next target. Only a survey with no usable coarse fits can widen.
    Only fine acceptance updates IDW, trusted history and linewidth feedback.
    single retains whole-window fitting and widening. Save every Completed
    attempt before analysis. Record all candidate rejections without inventing
    filenames for failed acquisition. Partial results stop without feedback.
    Invalid flux raises; unexpected acquisition/save/analysis errors propagate.
    """
    tracker, hint, predictor_model = _capture_frequency_prediction(carry, flux)
    prior = carry.seed if carry.last is None else carry.last
    tun = inputs.tun
    gain_clamped: list[GainClamp] = []
    gains = frequency_probe_gains(inputs, tracker, gain_clamped)
    first_stage: Literal["coarse", "fine", "single"] = (
        "coarse" if tun.frequency_mode == "two_step" else "single"
    )
    pending: list[
        tuple[Literal["coarse", "fine", "single"], float, FrequencyFit | None]
    ] = [(first_stage, widen, None) for widen in tun.qf_widen]
    scans: list[FrequencyScan] = []
    attempts: list[FrequencyCandidateAttempt] = []
    status, reason = "rejected", "No accepted frequency fit"
    coarse_file: str | None = None
    while pending:
        stage, widen, coarse_fit = pending.pop(0)
        if scope.cancelled:
            status, reason = "skipped", "Point cancelled"
            break
        cfg, scan = _plan_frequency_scan(
            inputs,
            prior,
            (stage, widen),
            hint if coarse_fit is None else coarse_fit.frequency_mhz,
            gains,
            len(scans) + 1,
        )
        scan, result = await acquire_frequency_scan(
            scope, cfg, scan=scan, inputs=inputs
        )
        scans.append(scan)
        if stage == "coarse":
            coarse_file = scan.file
            scan, fits, rejected = fit_coarse_candidates(
                result, scan, tun, tun.max_candidates - len(attempts)
            )
            scans[-1] = scan
            attempts.extend(rejected)
            reason = (
                result.reason
                if isinstance(result, Failed)
                else "No usable coarse candidates"
            )
            status = "rejected" if reason == "stopped" else "failed"
            if fits:
                _record_frequency_trace(fits[0], inputs, carry, plots, (flux, hint))
                # A fine failure tries the next saved target, never another survey.
                pending = [("fine", 1.0, fit) for fit in fits]
            elif len(attempts) >= tun.max_candidates or reason == "stopped":
                pending.clear()
            continue
        scan, fit = fit_frequency_scan(
            result, scan, tun, None if stage == "fine" else hint
        )
        scans[-1] = scan
        _record_frequency_trace(fit, inputs, carry, plots, (flux, hint))
        rejection = fit.reason if isinstance(fit, Failed) else None
        if coarse_fit is not None:
            scans[-1], confirmation = confirm_frequency_candidate(
                coarse_fit, coarse_file, scan, fit, tun, result
            )
            attempts.append(confirmation)
            rejection = confirmation.rejection_reason
        if isinstance(fit, Failed) or rejection is not None:
            status, reason = (
                _frequency_failure_status(fit, rejection),
                rejection or "No accepted frequency fit",
            )
            if reason == "stopped":
                break
            continue
        if _accept_frequency_result(fit, inputs, carry, plots, (flux, hint), scan.gain):
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
                tuple(attempts),
                gain_clamped=tuple(gain_clamped),
            )
        status, reason = "rejected", "No accepted frequency fit"
    if reason not in ("stopped", "Point cancelled"):
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
        tuple(attempts),
        gain_clamped=tuple(gain_clamped),
    )
