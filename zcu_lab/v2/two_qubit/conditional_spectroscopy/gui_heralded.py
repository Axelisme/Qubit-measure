"""Measure adapter for post-preparation heralded spectroscopy."""

from pathlib import Path
from typing import TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis, AnalyzeRequest
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from .core_heralded import (
    HeraldedSpectroscopyCfg,
    HeraldedSpectroscopyExp,
    HeraldedSpectroscopyResult,
)
from .form import spectroscopy_form

HeraldedSpectroscopyRun: TypeAlias = RunRecord[
    HeraldedSpectroscopyCfg, HeraldedSpectroscopyResult
]


def make_form(ctx: FormContext) -> Form[HeraldedSpectroscopyCfg]:
    """Seed the heralded form from the same fixed capture as ordinary spectroscopy."""
    return spectroscopy_form(ctx, HeraldedSpectroscopyCfg)


def run(
    cfg: HeraldedSpectroscopyCfg, *, context: RunContext
) -> HeraldedSpectroscopyRun:
    """Acquire the joint counts in context and retain the applied configuration."""
    return RunRecord(
        cfg=cfg, result=HeraldedSpectroscopyExp().run(cfg, context=context)
    )


def _analyze(
    req: AnalyzeRequest[HeraldedSpectroscopyRun, None], *, plots: Plots
) -> None:
    HeraldedSpectroscopyExp().analyze(req.run_result, None, plots=plots)


def _summary_lines(source: HeraldedSpectroscopyRun) -> tuple[str, ...]:
    cfg = source.cfg
    return () if cfg is None else (f"Reps×Rounds {cfg.reps}×{cfg.rounds}",)


DEFINITION = Adapter(
    spec=HeraldedSpectroscopyExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide_heralded.md"),
    analysis=Analysis(fn=_analyze, summary_lines=_summary_lines),
)
