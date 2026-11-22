"""Measure adapter for constructive phase."""

from collections.abc import Sequence
from pathlib import Path
from typing import TypeAlias

import numpy as np
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    ParameterWriteback,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.support.measure.writeback_helpers import module_writeback

from .core import (
    ConstructivePhaseAnalysis,
    ConstructivePhaseCfg,
    ConstructivePhaseExp,
    ConstructivePhaseResult,
)
from .form import make_form

ConstructivePhaseRun: TypeAlias = RunRecord[
    ConstructivePhaseCfg, ConstructivePhaseResult
]


def _analyze(
    req: AnalyzeRequest[ConstructivePhaseRun, None],
    *,
    plots: Plots,
) -> ConstructivePhaseAnalysis:
    """Analyze saved source cfg/counts, never the current Form or library."""
    return ConstructivePhaseExp().analyze(req.run_result, None, plots=plots)


def _get_writeback_items(
    req: WritebackRequest[ConstructivePhaseRun, ConstructivePhaseAnalysis],
) -> Sequence[WritebackItem]:
    """Propose only Qubit2 phase and its independent coupler slot from saved cfg.

    Qubit2.phase is source Qubit1.phase plus the identifiable maximum-rate relative
    phase. Preserve Qubit2 frequency, signed gain, waveform and all other fields.
    Missing Run/Load parameter capture raises ValueError. Missing cfg or
    unresolved phase returns no proposal. The capture supplies the coupler; it never reads current pulse values. No writes or atomic pair Apply.
    """
    parameters = req.ctx.parameters
    if parameters is None:
        raise ValueError("Constructive phase writeback requires captured parameters")
    snapshot = req.run_result.cfg
    phase = req.analyze_result.relative_phase_deg
    if snapshot is None or phase is None or not np.isfinite(phase):
        return []
    coupler = parameters.resolve_path("$coupler")
    target = f"{coupler}.pulses.cz_qubit2"
    key = parameters.library_key(target)
    return [
        module_writeback(
            snapshot.drive.qubit2,
            target=target,
            description="Constructive Qubit2 charge pulse, source Qubit1 phase reference",
            field_updates=(("phase", float(snapshot.drive.qubit1.phase) + phase),),
        ),
        ParameterWriteback(
            target_name=f"{coupler}.module.cz_qubit2",
            description="Qubit2 CZ pulse slot, independent Apply row",
            proposed_value=key,
        ),
    ]


def run(
    cfg: ConstructivePhaseCfg, *, context: RunContext
) -> RunRecord[ConstructivePhaseCfg, ConstructivePhaseResult]:
    """Run the experiment and retain its actual configuration."""
    return RunRecord(cfg=cfg, result=ConstructivePhaseExp().run(cfg, context=context))


def _summary_lines(source: ConstructivePhaseRun) -> tuple[str, ...]:
    cfg = source.cfg
    if cfg is None:
        return ()
    return (f"Reps×Rounds {cfg.reps}×{cfg.rounds}",)


DEFINITION = Adapter(
    spec=ConstructivePhaseExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_analyze, writeback=_get_writeback_items, summary_lines=_summary_lines
    ),
)
