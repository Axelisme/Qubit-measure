"""Measure GUI attachment for single-qubit Clifford randomized benchmarking."""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis, AnalyzeRequest
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.twotone.rb.core import RB_Exp, RB_Result, RBAnalysis, RBCfg

from .form import make_form

RBRunResult: TypeAlias = RunRecord[RBCfg, RB_Result]


def _RB_analyze(req: AnalyzeRequest[RBRunResult, None], *, plots: Plots) -> RBAnalysis:
    """Fit the saved run's mean IQ decay and publish its fit figure."""
    return RB_Exp().analyze(req.run_result, None, plots=plots)


def _RB_make_form(ctx: FormContext) -> Form[RBCfg]:
    """Build this adapter's definition using captured Open/Reset sources."""
    return make_form(ctx)


def _run(cfg: RBCfg, *, context: RunContext) -> RunRecord[RBCfg, RB_Result]:
    """Acquire a trace and retain the experiment configuration with its result."""
    return RunRecord(cfg=cfg, result=RB_Exp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=RB_Exp.AXES_SPEC,
    make_form=_RB_make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_RB_analyze),
)
