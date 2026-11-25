"""Measure GUI adapter for AC Stark Ramsey maps."""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
)
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.plotting.plots import Plots

from .core import (
    AcStarkRamseyAnalyzeOptions,
    AcStarkRamseyCfg,
    AcStarkRamseyExp,
    AcStarkRamseyResult,
)
from .form import make_form

AcStarkRamseyRunResult: TypeAlias = RunRecord[AcStarkRamseyCfg, AcStarkRamseyResult]


def _AcStarkRamsey_get_analyze_params(
    result: AcStarkRamseyRunResult, ctx: SessionEnv
) -> AcStarkRamseyAnalyzeOptions:
    """Seed the FFT center from the acquired cfg, or zero for cfg-free data."""
    del ctx
    return AcStarkRamseyAnalyzeOptions(
        detune=result.cfg.detune if result.cfg is not None else 0.0
    )


def _AcStarkRamsey_analyze(
    req: AnalyzeRequest[AcStarkRamseyRunResult, AcStarkRamseyAnalyzeOptions],
    *,
    plots: Plots,
) -> None:
    """Publish contrast/FFT maps, without fitting or proposing calibration."""
    AcStarkRamseyExp().analyze(
        req.run_result,
        req.analyze_params,
        plots=plots,
    )
    return


def _run_acstarkramsey(
    cfg: AcStarkRamseyCfg, *, context: RunContext
) -> RunRecord[AcStarkRamseyCfg, AcStarkRamseyResult]:
    """Acquire a trace and retain the experiment configuration with its result."""
    return RunRecord(cfg=cfg, result=AcStarkRamseyExp().run(cfg, context=context))


def _AcStarkRamsey_make_form(ctx: FormContext) -> Form[AcStarkRamseyCfg]:
    """Build the Ramsey form from captured Open/Reset sources."""
    return make_form(ctx)


def _ramsey_summary_lines(source: AcStarkRamseyRunResult) -> tuple[str, ...]:
    cfg = source.cfg
    return () if cfg is None else (f"Reps × Rounds: {cfg.reps} × {cfg.rounds}",)


DEFINITION = Adapter(
    spec=AcStarkRamseyExp.AXES_SPEC,
    make_form=_AcStarkRamsey_make_form,
    run=_run_acstarkramsey,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_AcStarkRamsey_analyze,
        options=_AcStarkRamsey_get_analyze_params,
        summary_lines=_ramsey_summary_lines,
    ),
)
