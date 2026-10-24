from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    AnalyzeResultBase,
    ParameterWriteback,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.support.measure.writeback_helpers import readout_dpm_writeback_items
from zcu_lab.v2.twotone.ro_optimize.auto_optimize.core import (
    AutoOptCfg,
    AutoOptExp,
    AutoOptResult,
)
from zcu_lab.v2.twotone.ro_optimize.auto_optimize.form import make_form

RoOptAutoRunResult: TypeAlias = RunRecord[AutoOptCfg, AutoOptResult]


@dataclass
class RoOptAutoAnalyzeResult(AnalyzeResultBase):
    best_freq: float
    best_gain: float
    best_length: float


def _RoOptAuto_make_form(ctx: FormContext) -> Form[AutoOptCfg]:
    """Build this adapter's definition using captured Open/Reset sources."""
    return make_form(ctx)


def _RoOptAuto_analyze(
    req: AnalyzeRequest[RoOptAutoRunResult, None],
    *,
    plots: Plots,
) -> RoOptAutoAnalyzeResult:
    result = AutoOptExp().analyze(req.run_result, None, plots=plots)
    return RoOptAutoAnalyzeResult(
        best_freq=result.best_freq,
        best_gain=result.best_gain,
        best_length=result.best_length,
    )


def _RoOptAuto_get_writeback_items(
    req: WritebackRequest[RoOptAutoRunResult, RoOptAutoAnalyzeResult],
) -> Sequence[WritebackItem]:
    result = req.analyze_result
    items: list[WritebackItem] = [
        ParameterWriteback(
            target_name="$qubit.ext.best_ro_freq",
            description="Optimal readout frequency (MHz)",
            proposed_value=result.best_freq,
        ),
        ParameterWriteback(
            target_name="$qubit.ext.best_ro_gain",
            description="Optimal readout gain (a.u.)",
            proposed_value=result.best_gain,
        ),
        ParameterWriteback(
            target_name="$qubit.ext.best_ro_length",
            description="Optimal readout length (us)",
            proposed_value=result.best_length,
        ),
    ]
    items.extend(
        readout_dpm_writeback_items(
            req.ctx,
            req.run_result.cfg,
            proposed={
                "best_ro_freq": result.best_freq,
                "best_ro_gain": result.best_gain,
                "best_ro_length": result.best_length,
            },
        )
    )
    return items


def _run(
    cfg: AutoOptCfg, *, context: RunContext
) -> RunRecord[AutoOptCfg, AutoOptResult]:
    """Acquire a trace and retain the experiment configuration with its result."""
    return RunRecord(cfg=cfg, result=AutoOptExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=AutoOptExp.AXES_SPEC,
    make_form=_RoOptAuto_make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_RoOptAuto_analyze, writeback=_RoOptAuto_get_writeback_items),
)
