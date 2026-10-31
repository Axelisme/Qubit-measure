"""Measure adapter for heralded four-branch excitation spectroscopy."""

from pathlib import Path
from typing import TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis, AnalyzeRequest
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from .core_heralded import (
    SpectroscopyHeraldedAnalysis,
    SpectroscopyHeraldedCfg,
    SpectroscopyHeraldedExp,
    SpectroscopyHeraldedResult,
)
from .form import spectroscopy_form

SpectroscopyHeraldedRun: TypeAlias = RunRecord[
    SpectroscopyHeraldedCfg, SpectroscopyHeraldedResult
]


def make_form(ctx: FormContext) -> Form[SpectroscopyHeraldedCfg]:
    """Capture shared probe inputs for the independently registered heralded version."""
    return spectroscopy_form(ctx, SpectroscopyHeraldedCfg)


def run(
    cfg: SpectroscopyHeraldedCfg, *, context: RunContext
) -> SpectroscopyHeraldedRun:
    """Run and retain the actual acquisition configuration."""
    return RunRecord(
        cfg=cfg, result=SpectroscopyHeraldedExp().run(cfg, context=context)
    )


def _analyze(
    req: AnalyzeRequest[SpectroscopyHeraldedRun, None], *, plots: Plots
) -> SpectroscopyHeraldedAnalysis:
    return SpectroscopyHeraldedExp().analyze(req.run_result, None, plots=plots)


DEFINITION = Adapter(
    spec=SpectroscopyHeraldedExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide_heralded.md"),
    analysis=Analysis(fn=_analyze),
)
