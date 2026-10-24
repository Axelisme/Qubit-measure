"""Measure GUI attachment for dispersive-shift spectroscopy."""

from pathlib import Path
from typing import TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis, AnalyzeRequest
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.twotone.dispersive.core import (
    DispersiveAnalysis,
    DispersiveAnalyzeOptions,
    DispersiveCfg,
    DispersiveExp,
    DispersiveResult,
)
from zcu_lab.v2.twotone.dispersive.form import make_form

DispersiveRunResult: TypeAlias = RunRecord[DispersiveCfg, DispersiveResult]


def _Dispersive_analyze(
    req: AnalyzeRequest[DispersiveRunResult, DispersiveAnalyzeOptions], *, plots: Plots
) -> DispersiveAnalysis:
    """Fit the source spectra, publish the fit figure, and return MHz values."""
    return DispersiveExp().analyze(
        req.run_result,
        req.analyze_params,
        plots=plots,
    )


def _Dispersive_make_form(ctx: FormContext) -> Form[DispersiveCfg]:
    """Build fresh inputs from the fixed Open/Reset capture."""
    return make_form(ctx)


def _run(
    cfg: DispersiveCfg, *, context: RunContext
) -> RunRecord[DispersiveCfg, DispersiveResult]:
    """Acquire a trace and retain the experiment configuration with its result."""
    return RunRecord(cfg=cfg, result=DispersiveExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=DispersiveExp.AXES_SPEC,
    make_form=_Dispersive_make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_Dispersive_analyze, options=lambda source, env: DispersiveAnalyzeOptions()
    ),
)
