from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import TypeAlias

from zcu_tools.cfg_model import TemplateDef
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis, AnalyzeRequest
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.twotone.ro_optimize.rpi_length.core import (
    RPiLengthAnalysis,
    RPiLengthCfg,
    RPiLengthExp,
    RPiLengthResult,
)
from zcu_lab.v2.twotone.ro_optimize.rpi_length.form import make_form

RPiLengthRunResult: TypeAlias = RunRecord[RPiLengthCfg, RPiLengthResult]


def _analyze(
    req: AnalyzeRequest[RPiLengthRunResult, None], *, plots: Plots
) -> RPiLengthAnalysis:
    return RPiLengthExp().analyze(req.run_result, None, plots=plots)


def _summary_lines(source: RPiLengthRunResult) -> tuple[str, ...]:
    cfg = source.cfg
    if cfg is None:
        return ()
    return (f"Reps × Rounds: {cfg.reps} × {cfg.rounds}",)


def _run(cfg: RPiLengthCfg, *, context: RunContext) -> RPiLengthRunResult:
    """Acquire the R-pi sums and keep the experiment configuration with them."""
    return RunRecord(cfg=cfg, result=RPiLengthExp().run(cfg, context=context))


def build_adapter(
    templates: Mapping[str, TemplateDef],
) -> Adapter[RPiLengthCfg, RPiLengthResult, None, RPiLengthAnalysis]:
    """Bind the registering Registry's templates for Open and Reset.

    Return an adapter without building a Form or inspecting the catalog.
    """
    # No writeback: the operator chooses the window from the score and its diagnostics.
    return Adapter(
        spec=RPiLengthExp.AXES_SPEC,
        make_form=partial(make_form, templates=templates),
        run=_run,
        guide=Path(__file__).with_name("guide.md"),
        analysis=Analysis(fn=_analyze, summary_lines=_summary_lines),
    )
