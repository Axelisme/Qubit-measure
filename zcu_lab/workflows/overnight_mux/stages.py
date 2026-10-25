"""One mux acquisition and independent participant fits per stage."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fitting import FitQuality
from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.workflows import Failed
from zcu_tools.program.v2 import SweepCfg

from zcu_lab.v2.mux.t1.core import MuxT1Cfg, MuxT1Exp, MuxT1ToneCfg
from zcu_lab.v2.mux.t2echo.core import MuxEchoToneCfg, MuxT2EchoCfg, MuxT2EchoExp
from zcu_lab.v2.support.lifetime_models import LifetimeModelRecord
from zcu_lab.v2.support.mux import MuxFitFailure
from zcu_lab.workflows.common.analysis import lifetime_rejection
from zcu_lab.workflows.common.calibration_results import StageStatus
from zcu_lab.workflows.common.gates import Gate, gate
from zcu_lab.workflows.common.saving import save_native
from zcu_lab.workflows.common.stages import StageEnv
from zcu_lab.workflows.common.summary import observe_operation
from zcu_lab.workflows.overnight_mux.pulses import FixedMuxPulses


@dataclass(frozen=True, kw_only=True)
class FitSettings:
    """Captured fit bounds, independent of mux pulses and timing.

    max_rel_err and max_fit_residual are strict dimensionless upper bounds.
    min_lifetime_points counts fitted delays within [0, lifetime_us].
    max_model_window_factor affects diagnostic models only, not acceptance.
    """

    max_rel_err: float
    max_fit_residual: float
    min_lifetime_points: int
    max_model_window_factor: float


@dataclass(frozen=True, kw_only=True)
class T1Settings(FitSettings):
    """Mux T1 averaging, shared delay policy and inherited fit acceptance bounds.

    reps/rounds are native acquisition averages. acquire_retry counts transient
    retries; earlystop_snr is the active tones' minimum completed-round SNR, or
    None to disable. uniform selects linear delays, otherwise the core
    concentrates the same window. sweep_factor times guess_us sets its endpoint.
    Relaxation is max(relax_min_us, relax_factor*guess_us), in us.
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
    """Mux Echo averaging, shared sweep factor and inherited fit bounds.

    reps/rounds are native acquisition averages. acquire_retry counts transient
    retries; earlystop_snr is the active tones' minimum completed-round SNR, or
    None to disable. sweep_factor times the Echo guess sets its linear endpoint.
    The main loop resolves shared relaxation separately in EchoTiming.
    """

    reps: int
    rounds: int
    acquire_retry: int
    earlystop_snr: float | None
    sweep_factor: float


@dataclass(frozen=True, kw_only=True)
class EchoTiming:
    """This round's longest-qubit Echo guess and resolved relaxation, in us.

    guess_us sets the shared sweep endpoint through EchoSettings.sweep_factor.
    relax_delay_us is the main loop's max-accepted-T1-or-fallback decision after
    applying its relaxation factor and floor. The stage does not re-decide it.
    """

    guess_us: float
    relax_delay_us: float


@dataclass(frozen=True, kw_only=True)
class QubitFitResult:
    """One original tone index's status, accepted lifetime/error in us and gates.

    index addresses the full cfg tone order, not the participant-list position.
    reason is None on acceptance, stopped on partial, otherwise the failure or
    failed comparisons. lifetime_us/error_us are None unless accepted. models
    holds diagnostic fits after primary-fit success; gates is empty until then.
    """

    index: int
    status: StageStatus
    reason: str | None = None
    lifetime_us: float | None = None
    error_us: float | None = None
    models: tuple[LifetimeModelRecord, ...] = ()
    gates: tuple[Gate, ...] = ()


@dataclass(frozen=True, kw_only=True)
class MuxT1StageResult:
    """T1 participant outcomes and saved t1.h5, or no file on acquisition failure.

    qubits retains acquisition order and original tone indices. Partial raw is
    saved with rejected/stopped outcomes, without fitting. Unexpected errors
    propagate without a result; already saved raw remains on disk.
    """

    qubits: tuple[QubitFitResult, ...]
    file: str | None = None


@dataclass(frozen=True, kw_only=True)
class MuxEchoStageResult:
    """Echo participant outcomes and t2echo.h5, absent on acquisition failure.

    qubits retains acquisition order and original tone indices. Partial raw is
    saved with rejected/stopped outcomes, without fitting. Unexpected errors
    propagate without a result; already saved raw remains on disk.
    """

    qubits: tuple[QubitFitResult, ...]
    file: str | None = None


def _acceptance(
    lifetime: float,
    error: float,
    quality: FitQuality | None,
    settings: FitSettings,
    name: str,
    times: NDArray[np.float64],
) -> tuple[str | None, tuple[Gate, ...]]:
    gates = (
        gate("positive_lifetime_us", ">", 0, lifetime),
        gate("nonnegative_error_us", ">=", 0, error),
        gate(
            "relative_error",
            "<",
            settings.max_rel_err,
            error / lifetime if lifetime > 0 else None,
        ),
        gate(
            "normalized_residual_rms",
            "<",
            settings.max_fit_residual,
            quality.normalized_residual_rms if quality is not None else None,
        ),
        gate(
            "invalid_quality_count",
            "==",
            0,
            len(quality.invalid) if quality is not None else None,
        ),
        gate(
            "points_within_lifetime",
            ">=",
            settings.min_lifetime_points,
            int(np.count_nonzero((times >= 0) & (times <= lifetime))),
        ),
    )
    rejection = lifetime_rejection(
        quality, times, lifetime, settings.min_lifetime_points
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


async def measure_mux_t1(
    scope: Scope,
    env: StageEnv,
    pulses: FixedMuxPulses,
    settings: T1Settings,
    *,
    points: int,
    guess_us: float,
) -> MuxT1StageResult:
    """Build mux T1 from fixed pulses, save raw and fit participants independently.

    pulses supplies full tone order, calibrated pi drives, readout, participant
    indices and optional reset. settings controls averaging, delay policy and fit
    gates. points is the shared delay count, at least five; guess_us is the
    positive longest-qubit estimate in us. The sweep ends at sweep_factor*guess_us;
    relaxation is max(relax_min_us, relax_factor*guess_us). Inputs remain unchanged.
    env supplies plots, save identity and summary observation. Failed/partial
    acquisitions return outcomes for every participant; expected per-tone fit
    failures leave others available. Other errors propagate. No point/history changes.
    """
    stop = settings.sweep_factor * guess_us
    cfg = MuxT1Cfg(
        uniform=settings.uniform,
        reps=settings.reps,
        rounds=settings.rounds,
        acquire_retry=settings.acquire_retry,
        earlystop_snr=settings.earlystop_snr,
        relax_delay=max(settings.relax_min_us, settings.relax_factor * guess_us),
        reset=pulses.reset,
        readout=pulses.readout,
        tones=[MuxT1ToneCfg(name=t.name, drive=t.drive) for t in pulses.tones],
        participants=list(pulses.participants),
        delays=SweepCfg(start=0.0, stop=stop, expts=points, step=stop / (points - 1)),
    )
    record = await scope.run(MuxT1Exp().run, cfg, name="Mux T1")
    observe_operation(env.summary, record)
    if isinstance(record, Failed):
        return MuxT1StageResult(
            qubits=tuple(
                QubitFitResult(index=index, status="failed", reason=record.reason)
                for index in cfg.participants
            )
        )
    save_native(
        record,
        scope.dir / "t1.h5",
        spec=MuxT1Exp.AXES_SPEC,
        snapshot=env.snapshot,
        identity=scope.identity,
    )
    if record.partial_result:
        return MuxT1StageResult(
            file="t1.h5",
            qubits=tuple(
                QubitFitResult(index=index, status="rejected", reason="stopped")
                for index in cfg.participants
            ),
        )
    # Native fit_i figures are temporary; Echo and later rounds reuse the names.
    fit_plots = env.plots.fork()
    try:
        fits = MuxT1Exp().analyze(
            record.value,
            MuxT1Exp.Options(skip=0, dual_exp=False),
            plots=fit_plots,
            max_model_window_factor=settings.max_model_window_factor,
        )
    finally:
        fit_plots.finish(present=False)
        fit_plots.release()
    qubits: list[QubitFitResult] = []
    for row, fit in enumerate(fits):
        if isinstance(fit, MuxFitFailure):
            qubits.append(
                QubitFitResult(index=fit.index, status="failed", reason=fit.reason)
            )
            continue
        quality = (
            fit.fit.fit_quality["fit"] if fit.fit.fit_quality is not None else None
        )
        reason, gates = _acceptance(
            fit.fit.t1,
            fit.fit.t1_err,
            quality,
            settings,
            "T1",
            record.value.result.times[np.isfinite(record.value.result.signals[row])],
        )
        qubits.append(
            QubitFitResult(
                index=fit.index,
                status="accepted" if reason is None else "rejected",
                reason=reason,
                lifetime_us=fit.fit.t1 if reason is None else None,
                error_us=fit.fit.t1_err if reason is None else None,
                models=fit.comparison.records if fit.comparison is not None else (),
                gates=gates,
            )
        )
    return MuxT1StageResult(file="t1.h5", qubits=tuple(qubits))


async def measure_mux_echo(
    scope: Scope,
    env: StageEnv,
    pulses: FixedMuxPulses,
    settings: EchoSettings,
    *,
    points: int,
    timing: EchoTiming,
) -> MuxEchoStageResult:
    """Build zero-detune mux Echo from fixed pi/pi2 pulses, save raw, fit and gate.

    pulses supplies full tone order, calibrated drives, readout, participants and
    optional reset. settings controls averaging, sweep factor and acceptance.
    points is the shared delay count, at least five. timing supplies the Echo
    guess and the main loop's resolved relaxation in us. The sweep ends at
    sweep_factor*timing.guess_us. Inputs and failure semantics otherwise match
    measure_mux_t1. Echo omits the first delay before fitting. No point record
    or history is changed; unexpected errors propagate.
    """
    stop = settings.sweep_factor * timing.guess_us
    cfg = MuxT2EchoCfg(
        reps=settings.reps,
        rounds=settings.rounds,
        acquire_retry=settings.acquire_retry,
        earlystop_snr=settings.earlystop_snr,
        relax_delay=timing.relax_delay_us,
        reset=pulses.reset,
        readout=pulses.readout,
        tones=[
            MuxEchoToneCfg(name=t.name, drive=t.pi2_pulse, pi_pulse=t.drive, detune=0.0)
            for t in pulses.tones
        ],
        participants=list(pulses.participants),
        delay=SweepCfg(start=0.0, stop=stop, expts=points, step=stop / (points - 1)),
    )
    record = await scope.run(MuxT2EchoExp().run, cfg, name="Mux Echo")
    observe_operation(env.summary, record)
    if isinstance(record, Failed):
        return MuxEchoStageResult(
            qubits=tuple(
                QubitFitResult(index=index, status="failed", reason=record.reason)
                for index in cfg.participants
            )
        )
    save_native(
        record,
        scope.dir / "t2echo.h5",
        spec=MuxT2EchoExp.AXES_SPEC,
        snapshot=env.snapshot,
        identity=scope.identity,
    )
    if record.partial_result:
        return MuxEchoStageResult(
            file="t2echo.h5",
            qubits=tuple(
                QubitFitResult(index=index, status="rejected", reason="stopped")
                for index in cfg.participants
            ),
        )
    fit_plots = env.plots.fork()
    try:
        fits = MuxT2EchoExp().analyze(
            record.value,
            MuxT2EchoExp.Options(fit_method="decay"),
            plots=fit_plots,
            max_model_window_factor=settings.max_model_window_factor,
        )
    finally:
        fit_plots.finish(present=False)
        fit_plots.release()
    qubits: list[QubitFitResult] = []
    for row, fit in enumerate(fits):
        if isinstance(fit, MuxFitFailure):
            qubits.append(
                QubitFitResult(index=fit.index, status="failed", reason=fit.reason)
            )
            continue
        reason, gates = _acceptance(
            fit.t2,
            fit.t2_err,
            fit.quality,
            settings,
            "T2 Echo",
            record.value.result.times[1:][
                np.isfinite(record.value.result.signals[row, 1:])
            ],
        )
        qubits.append(
            QubitFitResult(
                index=fit.index,
                status="accepted" if reason is None else "rejected",
                reason=reason,
                lifetime_us=fit.t2 if reason is None else None,
                error_us=fit.t2_err if reason is None else None,
                models=fit.comparison.records if fit.comparison is not None else (),
                gates=gates,
            )
        )
    return MuxEchoStageResult(file="t2echo.h5", qubits=tuple(qubits))
