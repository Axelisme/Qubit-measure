"""Build overnight experiments from fixed pulses, settings and lifetime guesses."""

from dataclasses import dataclass

import numpy as np
from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.workflows import Failed
from zcu_tools.program.v2 import SweepCfg

from zcu_lab.v2.support.lifetime_models import (
    LifetimeModelRecord,
    show_lifetime_comparison,
)
from zcu_lab.v2.twotone.time_domain.t1.core import T1Cfg, T1Exp
from zcu_lab.v2.twotone.time_domain.t2echo.core import T2EchoCfg, T2EchoExp
from zcu_lab.workflows.common.analysis import (
    DecayFit,
    analyze_t1,
    compare_decay_fit,
    lifetime_rejection,
)
from zcu_lab.workflows.common.calibration_results import StageStatus
from zcu_lab.workflows.common.gain import GainClamp, clamp_gain
from zcu_lab.workflows.common.gates import Gate, gate
from zcu_lab.workflows.common.stages import NativeOutput, StageEnv, acquire_saved
from zcu_lab.workflows.overnight.analysis import analyze_t2echo
from zcu_lab.workflows.overnight.pulses import FixedPulses


@dataclass(frozen=True, kw_only=True)
class FitSettings:
    """Acceptance and model diagnostics shared by this run's T1/Echo fits.

    max_rel_err and max_fit_residual are strict dimensionless upper bounds.
    min_lifetime_points counts fitted delays within [0, lifetime_us].
    max_model_window_factor only controls diagnostic model discard, not acceptance.
    """

    max_rel_err: float
    max_fit_residual: float
    min_lifetime_points: int
    max_model_window_factor: float


@dataclass(frozen=True, kw_only=True)
class T1Settings(FitSettings):
    """T1 averaging, delay policy and inherited fit acceptance bounds.

    reps/rounds are native acquisition averages. acquire_retry counts transient
    retries; earlystop_snr is a completed-round threshold, or None to disable.
    uniform selects linear delays, otherwise the core concentrates the same
    window. sweep_factor times guess_us gives its endpoint. relax_factor times
    guess_us gives relaxation, floored at relax_min_us. Delays are in us.
    """

    reps: int
    rounds: int
    acquire_retry: int
    earlystop_snr: float | None
    uniform: bool
    sweep_factor: float
    relax_factor: float
    relax_min_us: float


@dataclass(frozen=True, kw_only=True)
class EchoSettings(FitSettings):
    """Echo averaging, sweep factor and inherited fit acceptance bounds.

    reps/rounds are native acquisition averages. acquire_retry counts transient
    retries; earlystop_snr is a completed-round threshold, or None to disable.
    sweep_factor times the Echo guess gives its linear delay endpoint in us.
    The main loop resolves relaxation separately in EchoTiming.
    """

    reps: int
    rounds: int
    acquire_retry: int
    earlystop_snr: float | None
    sweep_factor: float


@dataclass(frozen=True, kw_only=True)
class EchoTiming:
    """This round's positive Echo guess and resolved relaxation, both in us.

    guess_us sets the sweep endpoint through EchoSettings.sweep_factor.
    relax_delay_us is the main loop's accepted-T1-or-fallback decision after
    applying its relaxation factor and floor. The stage does not re-decide it.
    """

    guess_us: float
    relax_delay_us: float


@dataclass(frozen=True, kw_only=True)
class T1Result:
    """T1 status and accepted lifetime/error in us, absent unless accepted.

    reason is None on acceptance, stopped on partial, otherwise the failure or
    failed gates. file references saved Completed raw, including partials and
    rejected fits; Failed acquisitions have no file. models holds both diagnostic
    fits after primary-fit success. gates holds numeric acceptance comparisons.
    Unexpected errors propagate without a result; saved raw remains on disk.
    """

    status: StageStatus
    reason: str | None = None
    lifetime_us: float | None = None
    error_us: float | None = None
    file: str | None = None
    models: tuple[LifetimeModelRecord, ...] = ()
    gates: tuple[Gate, ...] = ()


@dataclass(frozen=True, kw_only=True)
class EchoResult:
    """Echo status and accepted lifetime/error in us, absent unless accepted.

    gain_clamped records truncated pi/2 gain requests during cfg construction.

    reason is None on acceptance, stopped on partial, otherwise the failure or
    failed gates. file references saved Completed raw, including partials and
    rejected fits; Failed acquisitions have no file. models holds both diagnostic
    fits after primary-fit success. gates holds numeric acceptance comparisons.
    Unexpected errors propagate without a result; saved raw remains on disk.
    """

    status: StageStatus
    reason: str | None = None
    lifetime_us: float | None = None
    error_us: float | None = None
    file: str | None = None
    models: tuple[LifetimeModelRecord, ...] = ()
    gates: tuple[Gate, ...] = ()
    gain_clamped: tuple[GainClamp, ...] = ()


def _acceptance(
    fit: DecayFit, settings: FitSettings, name: str
) -> tuple[str | None, tuple[Gate, ...]]:
    gates = (
        gate("positive_lifetime_us", ">", 0, fit.lifetime_us),
        gate("nonnegative_error_us", ">=", 0, fit.error_us),
        gate(
            "relative_error",
            "<",
            settings.max_rel_err,
            fit.error_us / fit.lifetime_us if fit.lifetime_us > 0 else None,
        ),
        gate(
            "normalized_residual_rms",
            "<",
            settings.max_fit_residual,
            fit.quality.normalized_residual_rms,
        ),
        gate("invalid_quality_count", "==", 0, len(fit.quality.invalid)),
        gate(
            "points_within_lifetime",
            ">=",
            settings.min_lifetime_points,
            int(np.count_nonzero((fit.times >= 0) & (fit.times <= fit.lifetime_us))),
        ),
    )
    rejection = lifetime_rejection(
        fit.quality, fit.times, fit.lifetime_us, settings.min_lifetime_points
    )
    reasons = [rejection] if rejection is not None else []
    failed_comparison = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "==": "!="}
    reasons.extend(
        f"{item.name} {item.actual:.9g} {failed_comparison[item.comparison]} {item.threshold:.9g}"
        if item.actual is not None
        else f"{item.name} unavailable (requires {item.comparison} {item.threshold:.9g})"
        for item in gates
        if not item.passed
    )
    reason = f"{name} fit rejected: {'; '.join(reasons)}" if reasons else None
    return reason, gates


async def measure_t1(
    scope: Scope,
    env: StageEnv,
    pulses: FixedPulses,
    settings: T1Settings,
    *,
    points: int,
    guess_us: float,
) -> T1Result:
    """Build and acquire T1 from fixed pulses, save t1.h5, fit and gate.

    pulses supplies the plan's calibrated pi/readout. settings controls averaging,
    delay policy and acceptance. points is the delay count, at least five;
    guess_us is the positive lifetime estimate in us. The sweep ends at
    sweep_factor*guess_us; relaxation is max(relax_min_us, relax_factor*guess_us).
    Inputs remain unchanged. env supplies plots, save identity and summary.
    Partial returns rejected/stopped without fitting; acquisition/expected fit
    failure returns failed. Other errors propagate. No point or history changes.
    """
    stop = settings.sweep_factor * guess_us
    cfg = T1Cfg(
        uniform=settings.uniform,
        reps=settings.reps,
        rounds=settings.rounds,
        acquire_retry=settings.acquire_retry,
        earlystop_snr=settings.earlystop_snr,
        relax_delay=max(settings.relax_min_us, settings.relax_factor * guess_us),
        pi_pulse=pulses.pi,
        readout=pulses.readout,
        length=SweepCfg(start=0.0, stop=stop, expts=points, step=stop / (points - 1)),
    )
    acquisition = await acquire_saved(
        scope,
        env,
        T1Exp().run,
        cfg,
        name="T1",
        output=NativeOutput(filename="t1.h5", spec=T1Exp.AXES_SPEC),
    )
    if acquisition.result is None:
        return T1Result(
            status="rejected" if acquisition.status == "partial" else "failed",
            reason=acquisition.reason,
            file=acquisition.file,
        )
    fit = analyze_t1(acquisition.result)
    if isinstance(fit, Failed):
        return T1Result(status="failed", reason=fit.reason, file=acquisition.file)
    comparison = compare_decay_fit(
        fit, t1=True, max_window_factor=settings.max_model_window_factor
    )
    show_lifetime_comparison(
        env.plots, "t1-models", fit.times, fit.observed, comparison
    )
    reason, gates = _acceptance(fit, settings, "T1")
    return T1Result(
        status="accepted" if reason is None else "rejected",
        reason=reason,
        lifetime_us=fit.lifetime_us if reason is None else None,
        error_us=fit.error_us if reason is None else None,
        file=acquisition.file,
        models=comparison.records,
        gates=gates,
    )


async def measure_echo(
    scope: Scope,
    env: StageEnv,
    pulses: FixedPulses,
    settings: EchoSettings,
    *,
    points: int,
    timing: EchoTiming,
) -> EchoResult:
    """Build zero-detune Echo from fixed pulses, save t2echo.h5, fit and gate.

    pulses supplies pi/readout; pi/2 uses half the pi gain through clamp_gain.
    settings controls averaging, sweep factor, acceptance and model diagnostics.
    points is the delay count, at least five. timing supplies the Echo guess and
    the main loop's resolved relaxation in us. The sweep ends at
    sweep_factor*timing.guess_us. env supplies mechanical services. Partial
    returns rejected/stopped without fitting; acquisition/expected fit failure
    returns failed. Other errors propagate. No point or history is changed.
    """
    gain_clamped: list[GainClamp] = []
    stop = settings.sweep_factor * timing.guess_us
    cfg = T2EchoCfg(
        reps=settings.reps,
        rounds=settings.rounds,
        acquire_retry=settings.acquire_retry,
        earlystop_snr=settings.earlystop_snr,
        relax_delay=timing.relax_delay_us,
        pi_pulse=pulses.pi,
        pi2_pulse=pulses.pi.with_updates(
            gain=clamp_gain(pulses.pi.gain / 2, "echo.pi2", gain_clamped)
        ),
        readout=pulses.readout,
        length=SweepCfg(start=0.0, stop=stop, expts=points, step=stop / (points - 1)),
    )
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
            gain_clamped=tuple(gain_clamped),
        )
    fit = analyze_t2echo(acquisition.result)
    if isinstance(fit, Failed):
        return EchoResult(
            status="failed",
            reason=fit.reason,
            file=acquisition.file,
            gain_clamped=tuple(gain_clamped),
        )
    comparison = compare_decay_fit(
        fit, t1=False, max_window_factor=settings.max_model_window_factor
    )
    show_lifetime_comparison(
        env.plots, "t2echo-models", fit.times, fit.observed, comparison
    )
    reason, gates = _acceptance(fit, settings, "T2 Echo")
    return EchoResult(
        status="accepted" if reason is None else "rejected",
        reason=reason,
        lifetime_us=fit.lifetime_us if reason is None else None,
        error_us=fit.error_us if reason is None else None,
        file=acquisition.file,
        models=comparison.records,
        gates=gates,
        gain_clamped=tuple(gain_clamped),
    )
