"""Frequency calibration: coarse survey, candidate confirmation and feedback."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.workflows import Failed
from zcu_tools.program import shot_time_s
from zcu_tools.program.v2 import SweepCfg

from zcu_lab.v2.twotone.freq.core import FreqCfg, FreqResult
from zcu_lab.workflows.common.calibration_analysis import FrequencyFit
from zcu_lab.workflows.common.calibration_results import (
    FrequencyCandidateAttempt,
    FrequencyResult,
    FrequencyScan,
    StageStatus,
)
from zcu_lab.workflows.common.calibration_stage_types import (
    CalibrationPulses,
    FrequencyGuess,
    FrequencyStageResult,
)
from zcu_lab.workflows.common.calibration_tunables import CalibrationTunables
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
from zcu_lab.workflows.common.stages import StageEnv

if TYPE_CHECKING:
    from zcu_tools.simulate.fluxonium.physical_fit import FluxoniumModelSnapshot

    from zcu_lab.workflows.common.frequency_tracking import FrequencyTracker


def _frequency_cfg(
    pulses: CalibrationPulses,
    tun: CalibrationTunables,
    center: float,
    gain: float,
    span: float,
    *,
    gain_clamped: list[GainClamp] | None = None,
) -> FreqCfg:
    drive = pulses.probe.model_copy(deep=True)
    drive.set_param("gain", clamp_gain(gain, "probe", gain_clamped))
    ro = pulses.readout.model_copy(deep=True)
    ro.set_param("gain", clamp_gain(float(ro.gain), "readout", gain_clamped, lower=0.0))
    return FreqCfg(
        reps=tun.qf_reps,
        rounds=tun.qf_rounds,
        acquire_retry=tun.qf_acquire_retry,
        earlystop_snr=tun.qf_earlystop_snr,
        relax_delay=1.0,
        qub_pulse=drive,
        readout=ro,
        freq=SweepCfg(
            start=center - span / 2,
            stop=center + span / 2,
            expts=tun.qf_points,
            step=span / (tun.qf_points - 1),
        ),
    )


def estimate_frequency_time_s(
    pulses: CalibrationPulses, settings: CalibrationTunables
) -> float:
    """Return nominal spectroscopy shot seconds without acquiring or reading history.

    pulses supplies a scalar probe and retained pulse readout, with lengths and
    pre/post/trigger/ADC delays in us. settings supplies qf_points, reps, rounds
    and frequency_mode. single counts one scan; two_step counts one coarse and
    one fine scan. Frequency spans/gains do not affect duration. Use full rounds,
    no retries, early stop, widening or multiple candidates. Exclude acquisition
    overhead and flux ramp; hardware clock rounding is not modeled.
    Inputs are unchanged. Invalid cfg/timing raises ValueError; non-scalar module
    timing raises TypeError. The caller decides whether this stage is enabled.
    """
    cfg = _frequency_cfg(
        pulses,
        settings,
        float(pulses.probe.freq),
        float(pulses.probe.gain),
        settings.span_mhz,
    )
    probe_us = (
        float(cfg.qub_pulse.pre_delay)
        + float(cfg.qub_pulse.waveform.length)
        + float(cfg.qub_pulse.post_delay)
    )
    ro = pulses.readout
    readout_us = max(
        float(ro.trig_offset) + float(ro.ro_length),
        float(ro.pre_delay) + float(ro.waveform.length) + float(ro.post_delay),
    )
    scans = 2 if settings.frequency_mode == "two_step" else 1
    return scans * shot_time_s(
        rounds=cfg.rounds,
        reps=cfg.reps,
        shot_lengths_us=[probe_us + readout_us] * cfg.freq.expts,
        relax_delay_us=cfg.relax_delay,
    )


def _plan_frequency_scan(
    pulses: CalibrationPulses,
    tun: CalibrationTunables,
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
    span = (
        (tun.fine_span_mhz or tun.span_mhz)
        if stage == "fine"
        else tun.coarse_span_mhz * widen
        if stage == "coarse"
        else tun.span_mhz * widen
    )
    offset = 0.0 if stage == "fine" else tun.detune_center_mhz
    filename = (
        f"frequency-{index:02d}.h5"
        if stage == "single"
        else f"frequency-{stage}-{index:02d}.h5"
    )
    gain_clamped: list[GainClamp] = []
    cfg = _frequency_cfg(
        pulses, tun, center + offset, gain, span, gain_clamped=gain_clamped
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


def _frequency_failure_status(
    fit: FrequencyFit | Failed, reason: str | None
) -> StageStatus:
    """Separate expected confirmation/gate rejection from acquisition/fit failure."""
    if reason in ("stopped", "Selected peak is truncated") or (
        reason is not None and reason.startswith("False coarse peak:")
    ):
        return "rejected"
    return "failed" if isinstance(fit, Failed) else "rejected"


def _begin_point(
    tracker: FrequencyTracker, guess: FrequencyGuess
) -> tuple[float, FluxoniumModelSnapshot, float | None]:
    """Capture the point's predicted centre, model and matrix-element ratio."""
    if not math.isfinite(guess.point.flux):
        raise ValueError("Calibration flux must be finite")
    hint = tracker.begin_point(guess.point.flux, seed_flux=guess.seed_flux)
    return hint, tracker.model_snapshot(), tracker.m_ratio


def _coarse_outcome(
    result: FreqResult | Failed,
    scan: FrequencyScan,
    tun: CalibrationTunables,
    attempts: list[FrequencyCandidateAttempt],
) -> tuple[FrequencyScan, tuple[FrequencyFit, ...], StageStatus, str]:
    """Fit coarse targets, add their rejections, and name the survey failure."""
    scan, fits, rejected = fit_coarse_candidates(
        result, scan, tun, tun.max_candidates - len(attempts)
    )
    attempts.extend(rejected)
    reason = (
        result.reason if isinstance(result, Failed) else "No usable coarse candidates"
    )
    return scan, fits, "rejected" if reason == "stopped" else "failed", reason


async def calibrate_frequency(
    scope: Scope,
    env: StageEnv,
    pulses: CalibrationPulses,
    settings: CalibrationTunables,
    *,
    guess: FrequencyGuess,
    tracker: FrequencyTracker,
) -> FrequencyStageResult:
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
    hint, predictor_model, m_ratio = _begin_point(tracker, guess)
    gain_clamped: list[GainClamp] = []
    gains = frequency_probe_gains(pulses.probe, settings, tracker, gain_clamped)
    last_fit: FrequencyFit | None = None
    pending: list[
        tuple[Literal["coarse", "fine", "single"], float, FrequencyFit | None]
    ] = [
        ("coarse" if settings.frequency_mode == "two_step" else "single", widen, None)
        for widen in settings.qf_widen
    ]
    scans: list[FrequencyScan] = []
    attempts: list[FrequencyCandidateAttempt] = []
    status, reason = "rejected", "No accepted frequency fit"
    coarse_file, accepted_mhz = None, None
    while pending:
        stage, widen, coarse_fit = pending.pop(0)
        if scope.cancelled:
            status, reason = "skipped", "Point cancelled"
            break
        cfg, scan = _plan_frequency_scan(
            pulses,
            settings,
            (stage, widen),
            hint if coarse_fit is None else coarse_fit.frequency_mhz,
            gains,
            len(scans) + 1,
        )
        scan, result = await acquire_frequency_scan(
            scope, env, cfg, scan=scan, settings=settings
        )
        scans.append(scan)
        if stage == "coarse":
            coarse_file = scan.file
            scan, fits, status, reason = _coarse_outcome(
                result, scan, settings, attempts
            )
            scans[-1] = scan
            if fits:
                last_fit = fits[0]
                # A fine failure tries the next saved target, never another survey.
                pending = [("fine", 1.0, fit) for fit in fits]
            elif len(attempts) >= settings.max_candidates or reason == "stopped":
                pending.clear()
            continue
        scan, fit = fit_frequency_scan(
            result, scan, settings, None if stage == "fine" else hint
        )
        scans[-1] = scan
        if not isinstance(fit, Failed):
            last_fit = fit
        rejection = fit.reason if isinstance(fit, Failed) else None
        if coarse_fit is not None:
            scans[-1], confirmation = confirm_frequency_candidate(
                coarse_fit, coarse_file, scan, fit, settings, result
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
        if accept_frequency_fit(fit, fit.source, settings):
            tracker.finish_point(guess.point.flux, fit.frequency_mhz, tun=settings)
            observe_frequency_linewidth(
                tracker, fit, fit.source, scan.gain, guess.point.index, settings
            )
            status, reason, accepted_mhz = "accepted", None, fit.frequency_mhz
            break
        status, reason = "rejected", "No accepted frequency fit"
    if accepted_mhz is None and reason not in ("stopped", "Point cancelled"):
        tracker.finish_point(guess.point.flux, None, tun=settings)
    return FrequencyStageResult(
        record=FrequencyResult(
            status,
            reason,
            accepted_mhz,
            tuple(scan.file for scan in scans if scan.file is not None),
            tuple(scan.gain for scan in scans),
            hint,
            predictor_model,
            tracker.last_recovery,
            tuple(scans),
            tuple(attempts),
            gain_clamped=tuple(gain_clamped),
        ),
        fit=last_fit,
        m_ratio=m_ratio,
    )
