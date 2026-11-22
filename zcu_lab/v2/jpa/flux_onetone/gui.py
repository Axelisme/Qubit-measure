"""jpa/flux_onetone GUI adapter — JPA flux × readout-frequency 2D survey.

Connects Run and figures-only Analyze to ``OneToneFluxExp`` and supplies the
operator guide. Analyze does not propose parameter updates.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis, AnalyzeRequest
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.jpa.flux_onetone.core import (
    OneToneFluxCfg,
    OneToneFluxExp,
    OneToneFluxResult,
)
from zcu_lab.v2.jpa.flux_onetone.form import make_form

JpaFluxOneToneRunResult: TypeAlias = RunRecord[OneToneFluxCfg, OneToneFluxResult]


def run(
    cfg: OneToneFluxCfg, *, context: RunContext
) -> RunRecord[OneToneFluxCfg, OneToneFluxResult]:
    """Acquire with cfg and retain it as the analysis and save source."""
    return RunRecord(cfg=cfg, result=OneToneFluxExp().run(cfg, context=context))


def _analyze(
    req: AnalyzeRequest[JpaFluxOneToneRunResult, None], *, plots: Plots
) -> None:
    OneToneFluxExp().analyze(req.run_result, req.analyze_params, plots=plots)


def _summary_lines(source: JpaFluxOneToneRunResult) -> tuple[str, ...]:
    cfg = source.cfg
    return () if cfg is None else (f"Reps × Rounds: {cfg.reps} × {cfg.rounds}",)


DEFINITION = Adapter(
    spec=OneToneFluxExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze, summary_lines=_summary_lines),
)
