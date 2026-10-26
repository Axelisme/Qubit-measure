"""Fixed-duration gain Rabi stage used by Coherence and MIST."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fitting import FitQuality, compute_fit_quality, fit_rabi
from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.workflows import Failed
from zcu_tools.program import shot_time_s
from zcu_tools.program.v2 import SweepCfg

from zcu_lab.v2.twotone.rabi.amp_rabi.core import (
    AmpRabiCfg,
    AmpRabiExp,
    AmpRabiResult,
    rabi_signal2real,
)
from zcu_lab.workflows.common.analysis import fit_or_failed, validate_fit_trace
from zcu_lab.workflows.common.calibration_plots import (
    CalibrationTrace,
)
from zcu_lab.workflows.common.calibration_results import (
    AmpCalibrationResult,
    RabiResult,
)
from zcu_lab.workflows.common.calibration_stage_types import (
    CalibrationPulses,
    RabiGuess,
    RabiStageResult,
)
from zcu_lab.workflows.common.calibration_tunables import CalibrationTunables
from zcu_lab.workflows.common.frequency_tracking import trusted_curve
from zcu_lab.workflows.common.gain import GainClamp, clamp_gain
from zcu_lab.workflows.common.saving import save_native
from zcu_lab.workflows.common.stages import StageEnv
from zcu_lab.workflows.common.summary import observe_operation


@dataclass(frozen=True)
class AmpFit:
    """Unfiltered pi/pi2 gain estimates with marginal error and projected curves.

    Gains use DAC full-scale fractions in a.u. within [-1, 1]; errors use a.u. observed/fitted align with
    result.amps. quality contains normalized residual diagnostics.
    """

    pi_gain: float
    error: float
    pi2_gain: float
    observed: NDArray[np.float64]
    fitted: NDArray[np.float64]
    quality: FitQuality


def analyze_amp_rabi(result: AmpRabiResult) -> AmpFit | Failed:
    """Fit the complete gain trace with free phase, returning gains and errors.

    Use a nondecaying cosine at fixed duration. Free phase also permits tracking
    windows that do not include zero gain. Nonfinite IQ or expected fit failure
    returns Failed; structural errors propagate. No saving or acceptance occurs.
    """
    failure = validate_fit_trace(result.amps, result.signals)
    if failure is not None:
        return failure
    observed = rabi_signal2real(result.signals)
    fit = fit_or_failed(
        lambda: fit_rabi(result.amps, observed, decay=False, init_phase=None)
    )
    if isinstance(fit, Failed):
        return fit
    pi, error, pi2, _, _, _, fitted, _ = fit
    return AmpFit(
        pi,
        error,
        pi2,
        observed,
        fitted,
        compute_fit_quality(observed, fitted, {}, None),
    )


def _accepted(fit: AmpFit, stop: float, tun: CalibrationTunables) -> bool:
    residual = fit.quality.normalized_residual_rms
    return (
        math.isfinite(fit.pi_gain)
        and tun.amp_min_gain <= fit.pi_gain <= tun.amp_max_window_fraction * stop
        and math.isfinite(fit.error)
        and fit.error >= 0
        and fit.error / fit.pi_gain < tun.max_rabi_rel_err
        and residual is not None
        and residual < tun.max_fit_residual
        and trusted_curve(fit.observed, fit.fitted, tun.rabi_max_mean_residual_ratio)
    )


def estimate_amp_rabi_time_s(
    pulses: CalibrationPulses,
    settings: CalibrationTunables,
    *,
    pi_length_us: float,
    guess_us: float,
) -> float:
    """Return nominal AmpRabi shot seconds, or zero in len_only mode.

    pulses supplies scalar drive pre/post delays and retained readout timing in
    us. pi_length_us is the positive finite fixed waveform duration: the current
    LenRabi pi in len_then_amp, or the resolved target in amp_only. settings
    supplies rabi_points, reps/rounds and auto/fixed relaxation. guess_us is a
    positive finite caller-supplied T1 in us; auto relax applies its factor/floor,
    fixed uses rabi_relax_delay_us. No history is read. Gain windows do not change
    shot duration. Count full rounds without retries, early stop, frequency
    widening or multiple candidates. Exclude overhead/flux ramp and clock
    rounding. Inputs unchanged. Invalid counts/timing raise ValueError;
    non-scalar module timing raises TypeError. Caller decides point eligibility.
    """
    if settings.rabi_mode == "len_only":
        return 0.0
    if any(not math.isfinite(x) or x <= 0 for x in (pi_length_us, guess_us)):
        raise ValueError("AmpRabi pi duration and T1 guess must be positive and finite")
    drive = pulses.qub.model_copy(deep=True)
    drive.set_param("length", pi_length_us)
    pulse_us = (
        float(drive.pre_delay) + float(drive.waveform.length) + float(drive.post_delay)
    )
    ro = pulses.readout
    readout_us = max(
        float(ro.trig_offset) + float(ro.ro_length),
        float(ro.pre_delay) + float(ro.waveform.length) + float(ro.post_delay),
    )
    relax_us = (
        max(settings.rabi_relax_min_us, settings.rabi_relax_factor * guess_us)
        if settings.rabi_relax_mode == "auto"
        else settings.rabi_relax_delay_us
    )
    return shot_time_s(
        rounds=settings.rabi_rounds,
        reps=settings.rabi_reps,
        shot_lengths_us=[pulse_us + readout_us] * settings.rabi_points,
        relax_delay_us=relax_us,
    )


async def calibrate_amp_rabi(
    scope: Scope,
    env: StageEnv,
    pulses: CalibrationPulses,
    settings: CalibrationTunables,
    *,
    guess: RabiGuess,
    rabi: RabiResult,
) -> RabiStageResult:
    """Sweep gain at rabi.pi_length_us, preserving its current accepted frequency.

    Caller supplies an accepted length (Len fit or amp-only target). len_then_amp
    scans zero to min(1, 2.5 * Len gain). amp_only starts zero to amp_gain_max,
    then follows last accepted Amp pi gain +/- amp_gain_half_width, clipped to
    [0, amp_gain_max]. Native data save before fit; partial never updates history.
    Return combined stage acceptance and optional trusted pi2 at the same length.
    Amp failure cannot promote the Len pulse to downstream acceptance.
    """
    if rabi.frequency_mhz is None or rabi.pi_length_us is None:
        raise ValueError("AmpRabi requires current frequency and fixed pulse length")
    tun = settings
    drive = pulses.qub.model_copy(deep=True)
    drive.set_param("freq", rabi.frequency_mhz)
    drive.set_param("length", rabi.pi_length_us)
    length = float(drive.waveform.length)
    ro = pulses.readout.model_copy(deep=True)
    gain_clamped: list[GainClamp] = []
    ro.set_param("gain", clamp_gain(float(ro.gain), "readout", gain_clamped, lower=0.0))
    start = 0.0
    if tun.rabi_mode == "len_then_amp":
        if rabi.drive_gain is None:
            raise ValueError("AmpRabi after Len requires its acquisition gain")
        stop = 2.5 * rabi.drive_gain
    elif guess.last_amp_gain is None:
        stop = tun.amp_gain_max
    else:
        start = guess.last_amp_gain - tun.amp_gain_half_width
        stop = guess.last_amp_gain + tun.amp_gain_half_width
    start = clamp_gain(start, "amp.start", gain_clamped, lower=0.0)
    stop = clamp_gain(
        stop,
        "amp.stop",
        gain_clamped,
        lower=0.0,
        upper=1.0 if tun.rabi_mode == "len_then_amp" else tun.amp_gain_max,
    )
    cfg = AmpRabiCfg(
        reps=tun.rabi_reps,
        rounds=tun.rabi_rounds,
        acquire_retry=tun.rabi_acquire_retry,
        earlystop_snr=tun.rabi_earlystop_snr,
        relax_delay=max(tun.rabi_relax_min_us, tun.rabi_relax_factor * guess.relax_us)
        if tun.rabi_relax_mode == "auto"
        else tun.rabi_relax_delay_us,
        qub_pulse=drive,
        readout=ro,
        gain=SweepCfg(
            start=start,
            stop=stop,
            expts=tun.rabi_points,
            step=(stop - start) / (tun.rabi_points - 1),
        ),
    )
    outcome = await scope.run(AmpRabiExp().run, cfg, name="AmpRabi")
    observe_operation(env.summary, outcome)
    amp = AmpCalibrationResult("failed", None, length, gain_clamped=tuple(gain_clamped))
    trace = None
    accepted_gain = None
    pi2_gain = None
    if isinstance(outcome, Failed):
        amp = replace(amp, reason=outcome.reason)
    else:
        save_native(
            outcome,
            scope.dir / "amp-rabi.h5",
            spec=AmpRabiExp.AXES_SPEC,
            snapshot=env.snapshot,
            identity=scope.identity,
        )
        amp = replace(amp, file="amp-rabi.h5")
        if outcome.partial_result:
            amp = replace(amp, status="rejected", reason="stopped")
        else:
            amp, trace = _analyze_completed(outcome.value.result, amp, settings, guess)
    if amp.status == "accepted":
        accepted_gain = (
            None
            if amp.pi_gain is None
            else clamp_gain(amp.pi_gain, "pi", gain_clamped, lower=tun.amp_min_gain)
        )
        pi2_gain = (
            None
            if amp.pi2_gain is None
            else clamp_gain(amp.pi2_gain, "pi2", gain_clamped, lower=tun.amp_min_gain)
        )
        amp = replace(amp, gain_clamped=tuple(gain_clamped))
    return RabiStageResult(
        record=replace(
            rabi,
            status=amp.status,
            reason=amp.reason,
            pi_length_us=length if amp.status == "accepted" else None,
            pi2_length_us=length if pi2_gain is not None else None,
            drive_gain=accepted_gain,
            pi2_drive_gain=pi2_gain,
            file=rabi.file if tun.rabi_mode == "len_then_amp" else amp.file,
            amp=amp,
            len_status=rabi.status if tun.rabi_mode == "len_then_amp" else None,
        ),
        amp_trace=trace,
    )


def _analyze_completed(
    result: AmpRabiResult,
    amp: AmpCalibrationResult,
    settings: CalibrationTunables,
    guess: RabiGuess,
) -> tuple[AmpCalibrationResult, CalibrationTrace]:
    fit = analyze_amp_rabi(result)
    trace = CalibrationTrace(
        guess.point.index,
        guess.point.flux,
        result.amps,
        result.signals.real if isinstance(fit, Failed) else fit.observed,
    )
    if isinstance(fit, Failed):
        amp = replace(amp, reason=fit.reason)
    else:
        accepted = _accepted(fit, float(np.max(result.amps)), settings)
        pi2 = (
            fit.pi2_gain
            if (
                accepted
                and amp.length_us >= 0.03
                and math.isfinite(fit.pi2_gain)
                and settings.amp_min_gain <= fit.pi2_gain <= 1.0
            )
            else None
        )
        amp = replace(
            amp,
            status="accepted" if accepted else "rejected",
            reason=None if accepted else "Amp Rabi fit rejected",
            pi_gain=fit.pi_gain,
            pi_gain_error=fit.error,
            pi2_gain=pi2,
        )
        trace = replace(
            trace, fitted=fit.fitted, accepted=fit.pi_gain if accepted else None
        )
    return amp, trace
