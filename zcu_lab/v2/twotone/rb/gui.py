"""Measure GUI attachment for single-qubit Clifford randomized benchmarking."""

from __future__ import annotations

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
)
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.twotone.rb.core import RB_Exp, RB_Result, RBCfg

from .form import make_form

RBRunResult: TypeAlias = RunRecord[RBCfg, RB_Result]


@dataclass
class RBAnalyzeResult(AnalyzeResultBase):
    """Single-qubit error per Clifford and mean Clifford fidelity (unitless)."""

    epc: float
    fidelity: float


def _RB_make_form(ctx: FormContext) -> Form[RBCfg]:
    """Build this adapter's definition using captured Open/Reset sources."""
    return make_form(ctx)


def _RB_analyze(
    req: AnalyzeRequest[RBRunResult, None], *, plots: Plots
) -> RBAnalyzeResult:
    """Fit the saved run's mean IQ decay and publish its fit figure."""
    result = RB_Exp().analyze(req.run_result, None, plots=plots)
    return RBAnalyzeResult(epc=result.epc, fidelity=result.fidelity)


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
