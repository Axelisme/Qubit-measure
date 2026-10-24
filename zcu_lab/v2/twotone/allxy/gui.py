"""AllXY gate-error check adapter.

Runs the 21 standard AllXY gate pairs and reuses ``AllXY_Exp.analyze``, which
fits power and detuning errors; the summary reports them.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis, AnalyzeRequest
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.twotone.allxy.core import (
    AllXY_Exp,
    AllXY_Result,
    AllXYAnalysis,
    AllXYAnalyzeOptions,
    AllXYCfg,
)

from .form import make_form

AllXYRunResult: TypeAlias = RunRecord[AllXYCfg, AllXY_Result]


def _AllXY_analyze(
    req: AnalyzeRequest[AllXYRunResult, AllXYAnalyzeOptions], *, plots: Plots
) -> AllXYAnalysis:
    return AllXY_Exp().analyze(
        req.run_result,
        req.analyze_params,
        plots=plots,
    )


def _AllXY_make_form(ctx: FormContext) -> Form[AllXYCfg]:
    """Build this adapter's definition using captured Open/Reset sources."""
    return make_form(ctx)


def _run(cfg: AllXYCfg, *, context: RunContext) -> RunRecord[AllXYCfg, AllXY_Result]:
    """Acquire a trace and retain the experiment configuration with its result."""
    return RunRecord(cfg=cfg, result=AllXY_Exp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=AllXY_Exp.AXES_SPEC,
    make_form=_AllXY_make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_AllXY_analyze, options=lambda source, env: AllXYAnalyzeOptions()
    ),
)
