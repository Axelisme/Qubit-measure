"""One mux acquisition and independent participant fits per stage."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fitting import FitQuality
from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.workflows import Failed

from zcu_lab.v2.mux.t1.core import MuxT1Cfg, MuxT1Exp
from zcu_lab.v2.mux.t2echo.core import MuxT2EchoCfg, MuxT2EchoExp
from zcu_lab.v2.support.lifetime_models import LifetimeModelRecord
from zcu_lab.v2.support.mux import MuxFitFailure
from zcu_lab.workflows.common.analysis import lifetime_rejection
from zcu_lab.workflows.common.calibration_results import StageStatus
from zcu_lab.workflows.common.gates import Gate, gate
from zcu_lab.workflows.common.saving import save_native
from zcu_lab.workflows.common.stages import StageEnv
from zcu_lab.workflows.common.summary import observe_operation


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


async def run_mux_t1_stage(
    scope: Scope, env: StageEnv, cfg: MuxT1Cfg, settings: FitSettings
) -> MuxT1StageResult:
    """Acquire fixed mux T1 cfg, save raw and fit each participant independently.

    cfg supplies full tone order, participant indices, delay grid and averaging.
    settings supplies fit gates and model diagnostics. env supplies workflow
    plots, historical save identity and summary observation. Failed/partial
    acquisitions return an outcome for every participant; expected per-tone fit
    failures leave other fits available. Unexpected errors propagate. No point
    record, cancellation state or accepted history is changed.
    """
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


async def run_mux_echo_stage(
    scope: Scope, env: StageEnv, cfg: MuxT2EchoCfg, settings: FitSettings
) -> MuxEchoStageResult:
    """Acquire fixed mux Echo cfg, save raw and fit zero-detune participant decays.

    Inputs and failure semantics match run_mux_t1_stage. Echo omits the first
    delay sample before fitting and applying gates. Each result retains its
    original tone index; no point record or accepted history is changed.
    """
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
