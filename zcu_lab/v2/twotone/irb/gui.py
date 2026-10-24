"""Measure GUI attachment for paired reference/interleaved randomized benchmarking."""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis, AnalyzeRequest
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.twotone.irb.analysis import IRBAnalysis
from zcu_lab.v2.twotone.irb.core import IRB_Exp, IRBCfg, IRBResult

from .form import make_form

IRBRunResult: TypeAlias = RunRecord[IRBCfg, IRBResult]


def _IRB_analyze(
    req: AnalyzeRequest[IRBRunResult, None], *, plots: Plots
) -> IRBAnalysis:
    """Fit the saved run's mean IQ decay and publish its fit figure."""
    return IRB_Exp().analyze(req.run_result, None, plots=plots)


def _IRB_make_form(ctx: FormContext) -> Form[IRBCfg]:
    """Build this adapter's definition using captured Open/Reset sources."""
    return make_form(ctx)


def _run(cfg: IRBCfg, *, context: RunContext) -> RunRecord[IRBCfg, IRBResult]:
    """Acquire a trace and retain the experiment configuration with its result."""
    return RunRecord(cfg=cfg, result=IRB_Exp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=IRB_Exp.AXES_SPEC,
    make_form=_IRB_make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_IRB_analyze),
)
