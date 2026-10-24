"""Measure GUI attachment for dispersive-shift spectroscopy."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    AnalyzeResultBase,
    ParamMeta,
)
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.twotone.dispersive.core import (
    DispersiveAnalyzeOptions,
    DispersiveCfg,
    DispersiveExp,
    DispersiveResult,
)
from zcu_lab.v2.twotone.dispersive.form import make_form

DispersiveRunResult: TypeAlias = RunRecord[DispersiveCfg, DispersiveResult]


@dataclass
class DispersiveAnalyzeParams:
    """Choose whether both resonance fits allow an amplitude-background slope."""

    fit_bg_amp_slope: Annotated[
        bool, ParamMeta(label="Fit background amplitude slope")
    ] = False


@dataclass
class DispersiveAnalyzeResult(AnalyzeResultBase):
    """Half the absolute resonance separation and mean FWHM, both in MHz."""

    chi: float
    avg_fwhm: float


def _Dispersive_make_form(ctx: FormContext) -> Form[DispersiveCfg]:
    """Build fresh inputs from the fixed Open/Reset capture."""
    return make_form(ctx)


def _Dispersive_analyze(
    req: AnalyzeRequest[DispersiveRunResult, DispersiveAnalyzeParams],
    *,
    plots: Plots,
) -> DispersiveAnalyzeResult:
    """Fit the source spectra, publish the fit figure, and return MHz values."""
    analysis = DispersiveExp().analyze(
        req.run_result,
        DispersiveAnalyzeOptions(fit_bg_amp_slope=req.analyze_params.fit_bg_amp_slope),
        plots=plots,
    )
    return DispersiveAnalyzeResult(chi=analysis.chi, avg_fwhm=analysis.avg_fwhm)


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
        fn=_Dispersive_analyze, options=lambda source, env: DispersiveAnalyzeParams()
    ),
)
