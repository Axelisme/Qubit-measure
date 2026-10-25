"""Fixed-duration gain Rabi stage used by Coherence and MIST."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fitting import FitQuality, compute_fit_quality, fit_rabi
from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.workflows import Failed
from zcu_tools.plotting.plots import Plots
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
    draw_rabi,
    retain_trace,
)
from zcu_lab.workflows.common.calibration_results import (
    AmpCalibrationResult,
    RabiResult,
)
from zcu_lab.workflows.common.frequency_tracking import trusted_curve
from zcu_lab.workflows.common.saving import save_native

if TYPE_CHECKING:
    from zcu_lab.workflows.common.calibration import CalCarry, CalibrationInputs


@dataclass(frozen=True)
class AmpFit:
    """Unfiltered pi/pi2 gain estimates with marginal error and projected curves.

    Gains/error use native normalized gain units. observed/fitted align with
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


def _accepted(fit: AmpFit, stop: float, inputs: CalibrationInputs) -> bool:
    tun = inputs.tun
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


async def calibrate_amp_rabi(
    scope: Scope,
    *,
    plots: Plots,
    inputs: CalibrationInputs,
    carry: CalCarry,
    rabi: RabiResult,
    t1_estimate_us: float,
) -> RabiResult:
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
    tun = inputs.tun
    prior = carry.seed if carry.last is None else carry.last
    drive = inputs.qub_pulse.model_copy(deep=True)
    drive.set_param("freq", rabi.frequency_mhz)
    drive.set_param("length", rabi.pi_length_us)
    length = float(drive.waveform.length)
    ro = inputs.readout.model_copy(deep=True)
    ro.set_param("freq", prior.readout_freq_mhz)
    ro.set_param("gain", prior.readout_gain)
    start = 0.0
    if tun.rabi_mode == "len_then_amp":
        if rabi.drive_gain is None:
            raise ValueError("AmpRabi after Len requires its acquisition gain")
        stop = min(1.0, 2.5 * rabi.drive_gain)
    elif carry.last_amp_gain is None:
        stop = tun.amp_gain_max
    else:
        start = max(0.0, carry.last_amp_gain - tun.amp_gain_half_width)
        stop = min(tun.amp_gain_max, carry.last_amp_gain + tun.amp_gain_half_width)
    cfg = AmpRabiCfg(
        reps=tun.rabi_reps,
        rounds=tun.rabi_rounds,
        acquire_retry=tun.rabi_acquire_retry,
        earlystop_snr=tun.rabi_earlystop_snr,
        relax_delay=max(tun.rabi_relax_min_us, tun.rabi_relax_factor * t1_estimate_us)
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
    amp = AmpCalibrationResult("failed", None, length)
    accepted_gain = None
    pi2_gain = None
    if isinstance(outcome, Failed):
        amp = replace(amp, reason=outcome.reason)
    else:
        save_native(
            outcome,
            inputs.directory / "amp-rabi.h5",
            spec=AmpRabiExp.AXES_SPEC,
            snapshot=inputs.snapshot,
            identity=scope.identity,
        )
        amp = replace(amp, file="amp-rabi.h5")
        if outcome.partial_result:
            amp = replace(amp, status="rejected", reason="stopped")
        else:
            amp = _analyze_completed(outcome.value.result, amp, inputs, carry, plots)
    if amp.status == "accepted":
        accepted_gain = amp.pi_gain
        pi2_gain = amp.pi2_gain
        carry.last_amp_gain = accepted_gain
        carry.last = replace(
            prior,
            f01_mhz=rabi.frequency_mhz,
            pi_length_us=length,
            drive_gain=accepted_gain,
        )
    return replace(
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
    )


def _analyze_completed(
    result: AmpRabiResult,
    amp: AmpCalibrationResult,
    inputs: CalibrationInputs,
    carry: CalCarry,
    plots: Plots,
) -> AmpCalibrationResult:
    fit = analyze_amp_rabi(result)
    trace = CalibrationTrace(
        inputs.point_index,
        inputs.flux,
        result.amps,
        result.signals.real if isinstance(fit, Failed) else fit.observed,
    )
    retain_trace(carry.amp_traces, trace)
    if isinstance(fit, Failed):
        amp = replace(amp, reason=fit.reason)
    else:
        accepted = _accepted(fit, float(np.max(result.amps)), inputs)
        pi2 = (
            fit.pi2_gain
            if (
                accepted
                and amp.length_us >= 0.03
                and math.isfinite(fit.pi2_gain)
                and inputs.tun.amp_min_gain <= fit.pi2_gain <= 1.0
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
        carry.amp_traces[-1] = replace(
            trace, fitted=fit.fitted, accepted=fit.pi_gain if accepted else None
        )
    draw_rabi(
        plots,
        carry.amp_traces,
        flux_unit=inputs.flux_unit,
        directory=inputs.directory,
        coordinate="gain",
    )
    return amp
