"""Measure Hahn-echo coherence from current-point pulses and independent timing."""

from dataclasses import dataclass
from typing import Literal

from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.workflows import Failed

from zcu_lab.v2.support.lifetime_models import LifetimeModelRecord
from zcu_lab.v2.twotone.time_domain.t2echo.core import T2EchoCfg, T2EchoExp
from zcu_lab.workflows.coherence_fluxdep.analysis import analyze_t2echo_fluxdep
from zcu_lab.workflows.coherence_fluxdep.pulses import CalibratedPulses
from zcu_lab.workflows.coherence_fluxdep.stages.t2_support import (
    T2Guess,
    decay_plot,
    delay_sweep,
    draw_decay_fit,
    relax_delay,
    t2_gates,
    t2_rejection_reason,
)
from zcu_lab.workflows.coherence_fluxdep.t2 import T2EchoTunables
from zcu_lab.workflows.common.analysis import compare_decay_fit
from zcu_lab.workflows.common.calibration_results import StageStatus
from zcu_lab.workflows.common.gain import GainClamp, clamp_gain
from zcu_lab.workflows.common.gates import Gate
from zcu_lab.workflows.common.stages import NativeOutput, StageEnv, acquire_saved


@dataclass(frozen=True, kw_only=True)
class EchoResult:
    """Independent echo outcome; construct with keywords, never update history.

    kind is t2echo. status is accepted, rejected, failed or skipped. reason is
    None on acceptance, otherwise the dependency, acquisition or fit diagnosis.
    lifetime_us/error_us contain accepted microseconds only, otherwise None.
    file is t2echo.h5 after saving any Completed, including partial/rejected;
    acquisition failures and dependency skips have no file. detune_mhz is None
    for Echo, preserving the existing record field.
    models contains both diagnostic fits; gates lists the existing acceptance
    comparisons. Both are empty before primary-fit success. gain_clamped holds
    truncated DAC requests from cfg construction, including failed acquisitions.
    Partial saves raw then returns rejected/stopped without fitting. Expected
    fit failure returns failed with its file; unexpected errors propagate.
    """

    kind: Literal["t2echo"] = "t2echo"
    status: StageStatus
    reason: str | None = None
    lifetime_us: float | None = None
    error_us: float | None = None
    file: str | None = None
    detune_mhz: float | None = None
    models: tuple[LifetimeModelRecord, ...] = ()
    gates: tuple[Gate, ...] = ()
    gain_clamped: tuple[GainClamp, ...] = ()


def _cfg(
    pulses: CalibratedPulses,
    settings: T2EchoTunables,
    *,
    points: int,
    guess: T2Guess,
) -> tuple[T2EchoCfg, tuple[GainClamp, ...]]:
    if pulses.pi2 is None:
        raise ValueError("Echo requires this point's accepted pi/pi2 pair")
    sweep = delay_sweep(settings, points=points, guess_us=guess.lifetime_us)
    gain_clamped: list[GainClamp] = []
    pi = pulses.pi.model_copy(deep=True)
    pi.set_param("gain", clamp_gain(float(pi.gain), "pi", gain_clamped))
    pi2 = pulses.pi2.model_copy(deep=True)
    pi2.set_param("gain", clamp_gain(float(pi2.gain), "pi2", gain_clamped))
    readout = pulses.readout.model_copy(deep=True)
    readout.set_param(
        "gain", clamp_gain(float(readout.gain), "readout", gain_clamped, lower=0.0)
    )
    return T2EchoCfg(
        reps=settings.reps,
        rounds=settings.rounds,
        acquire_retry=settings.acquire_retry,
        earlystop_snr=settings.earlystop_snr,
        relax_delay=relax_delay(settings, guess.t1_us),
        detune=settings.detune_ratio / sweep.step,
        pi_pulse=pi,
        pi2_pulse=pi2,
        readout=readout,
        length=sweep,
    ), tuple(gain_clamped)


async def measure_echo(
    scope: Scope,
    env: StageEnv,
    pulses: CalibratedPulses,
    settings: T2EchoTunables,
    *,
    points: int,
    guess: T2Guess,
) -> EchoResult:
    """Acquire echo, save raw, fit and draw both models using existing gates.

    pulses carries this point's accepted pi/pi2 pair and retained readout.
    settings controls averaging, window, detuning and minimum fitted points.
    points is the delay count. guess holds this stage's positive finite history
    estimate and the T1 estimate for relaxation, both in us.
    Auto timing caps sweep_factor*guess.lifetime_us and floors relax_factor*T1; fixed
    timing remains explicit. Detune is detune_ratio divided by the delay step.
    env supplies plots, snapshot and optional summary. Inputs stay unchanged.
    Save every Completed before fitting, including partials. Partial returns
    rejected/stopped; acquisition and expected fit failures return failed.
    Unexpected errors propagate. No point record or accepted history is changed.
    Nonzero detune selects a fringe fit; zero selects pure decay.
    """
    cfg, gain_clamped = _cfg(
        pulses,
        settings,
        points=points,
        guess=guess,
    )
    plot = decay_plot(env.plots, "t2echo")
    acquisition = await acquire_saved(
        scope,
        env,
        T2EchoExp().run,
        cfg,
        name="Echo",
        output=NativeOutput(filename="t2echo.h5", spec=T2EchoExp.AXES_SPEC),
    )
    if acquisition.result is None:
        return EchoResult(
            status="rejected" if acquisition.status == "partial" else "failed",
            reason=acquisition.reason,
            file=acquisition.file,
            gain_clamped=gain_clamped,
        )
    plot.update(acquisition.result.times, acquisition.result.signals.real)
    fit = analyze_t2echo_fluxdep(acquisition.result, fit_fringe=cfg.detune != 0.0)
    if isinstance(fit, Failed):
        return EchoResult(
            status="failed",
            reason=fit.reason,
            file=acquisition.file,
            gain_clamped=gain_clamped,
        )
    comparison = compare_decay_fit(
        fit,
        t1=False,
        fringe=cfg.detune != 0.0,
        max_window_factor=settings.max_model_window_factor,
    )
    gates = t2_gates(fit, settings.min_lifetime_points)
    reason = t2_rejection_reason(fit, settings.min_lifetime_points, gates)
    draw_decay_fit(env.plots, "t2echo", fit, comparison, reason)
    if reason is not None:
        return EchoResult(
            status="rejected",
            reason=reason,
            file=acquisition.file,
            models=comparison.records,
            gates=gates,
            gain_clamped=gain_clamped,
        )
    return EchoResult(
        status="accepted",
        lifetime_us=fit.lifetime_us,
        error_us=fit.error_us,
        file=acquisition.file,
        models=comparison.records,
        gates=gates,
        gain_clamped=gain_clamped,
    )
