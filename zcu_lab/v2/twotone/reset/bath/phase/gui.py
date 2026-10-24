from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TypeAlias

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
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.support.measure.writeback_helpers import module_writeback
from zcu_lab.v2.twotone.reset.bath.phase.core import (
    PhaseAnalysis,
    PhaseCfg,
    PhaseExp,
    PhaseResult,
)
from zcu_lab.v2.twotone.reset.bath.phase.form import make_form

BathPhaseRunResult: TypeAlias = RunRecord[PhaseCfg, PhaseResult]


def _BathPhase_analyze(
    req: AnalyzeRequest[BathPhaseRunResult, None], *, plots: Plots
) -> PhaseAnalysis:
    return PhaseExp().analyze(req.run_result, None, plots=plots)


def _BathPhase_get_writeback_items(
    req: WritebackRequest[BathPhaseRunResult, PhaseAnalysis],
) -> Sequence[WritebackItem]:
    """Propose ext phases, independent tested variants and a ground slot.

    None cfg or capture retains only scalars. Each variant copies this
    Run's tested_reset and changes only pi2_cfg.phase. Capture resolution
    and materialization failures propagate; ext is not reread.
    """
    result = req.analyze_result
    items: list[WritebackItem] = [
        ParameterWriteback(
            target_name="$qubit.ext.reset_bath_max_phase",
            description="Bath-reset pi/2 phase for ground reset (deg)",
            proposed_value=result.max_phase,
        ),
        ParameterWriteback(
            target_name="$qubit.ext.reset_bath_min_phase",
            description="Bath-reset pi/2 phase for excited reset (deg)",
            proposed_value=result.min_phase,
        ),
    ]

    cfg = req.run_result.cfg
    parameters = req.ctx.parameters
    if cfg is None or parameters is None:
        return items

    ground_root = parameters.resolve_path("$qubit.reset.reset_bath")
    for target, phase, description in (
        ("$qubit.reset.reset_bath", result.max_phase, "Bath reset to ground"),
        ("$qubit.reset.reset_bath_e", result.min_phase, "Bath reset to excited"),
    ):
        items.append(
            module_writeback(
                cfg.tested_reset,
                target=target,
                description=description,
                field_updates=(("pi2_cfg.phase", phase),),
            )
        )
    items.append(
        ParameterWriteback(
            target_name="$qubit.module.reset",
            description="Select bath reset for downstream experiments",
            proposed_value=ground_root,
        )
    )

    return items


def _BathPhase_make_form(ctx: FormContext) -> Form[PhaseCfg]:
    """Build this adapter's definition using captured Open/Reset sources."""
    return make_form(ctx)


def _run(cfg: PhaseCfg, *, context: RunContext) -> RunRecord[PhaseCfg, PhaseResult]:
    """Acquire a trace and retain the experiment configuration with its result."""
    return RunRecord(cfg=cfg, result=PhaseExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=PhaseExp.AXES_SPEC,
    make_form=_BathPhase_make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_BathPhase_analyze, writeback=_BathPhase_get_writeback_items),
)
