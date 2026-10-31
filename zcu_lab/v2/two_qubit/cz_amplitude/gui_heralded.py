"""GUI attachment for ground-ground heralded CZ amplitude calibration."""

from collections.abc import Sequence
from pathlib import Path
from typing import TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter, Analysis, AnalyzeRequest, WritebackItem, WritebackRequest,
)
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from .core_heralded import (
    CZAmplitudeHeraldedAnalysis, CZAmplitudeHeraldedCfg,
    CZAmplitudeHeraldedExp, CZAmplitudeHeraldedResult,
)
from .form import amplitude_form
from .gui import amplitude_writeback

CZAmplitudeHeraldedRun: TypeAlias = RunRecord[
    CZAmplitudeHeraldedCfg, CZAmplitudeHeraldedResult
]


def make_form(ctx: FormContext) -> Form[CZAmplitudeHeraldedCfg]:
    """Capture roles and seed editable heralded CZ amplitude inputs."""
    return amplitude_form(ctx, CZAmplitudeHeraldedCfg)


def run(
    cfg: CZAmplitudeHeraldedCfg, *, context: RunContext
) -> CZAmplitudeHeraldedRun:
    """Run the heralded experiment and retain its actual configuration."""
    return RunRecord(cfg=cfg, result=CZAmplitudeHeraldedExp().run(cfg, context=context))


def _analyze(
    req: AnalyzeRequest[CZAmplitudeHeraldedRun, None], *, plots: Plots
) -> CZAmplitudeHeraldedAnalysis:
    return CZAmplitudeHeraldedExp().analyze(req.run_result, None, plots=plots)


def _get_writeback_items(
    req: WritebackRequest[CZAmplitudeHeraldedRun, CZAmplitudeHeraldedAnalysis],
) -> Sequence[WritebackItem]:
    return amplitude_writeback(
        req.run_result.cfg, req.analyze_result.terminal.fit, req.ctx.parameters
    )


DEFINITION = Adapter(
    spec=CZAmplitudeHeraldedExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide_heralded.md"),
    analysis=Analysis(fn=_analyze, writeback=_get_writeback_items),
)
