from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis, AnalyzeRequest
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.twotone.ro_optimize.rpi_power.core import (
    RPiPowerAnalysis,
    RPiPowerCfg,
    RPiPowerExp,
    RPiPowerResult,
)
from zcu_lab.v2.twotone.ro_optimize.rpi_power.form import make_form

RPiPowerRunResult: TypeAlias = RunRecord[RPiPowerCfg, RPiPowerResult]


def _analyze(
    req: AnalyzeRequest[RPiPowerRunResult, None], *, plots: Plots
) -> RPiPowerAnalysis:
    return RPiPowerExp().analyze(req.run_result, None, plots=plots)


def _summary_lines(source: RPiPowerRunResult) -> tuple[str, ...]:
    cfg = source.cfg
    if cfg is None:
        return ()
    return (f"Reps × Rounds: {cfg.reps} × {cfg.rounds}",)


def _run(cfg: RPiPowerCfg, *, context: RunContext) -> RPiPowerRunResult:
    """Acquire the R-pi sums and keep the experiment configuration with them."""
    return RunRecord(cfg=cfg, result=RPiPowerExp().run(cfg, context=context))


# No writeback: the operator chooses the gain from the score and its diagnostics.
DEFINITION = Adapter(
    spec=RPiPowerExp.AXES_SPEC,
    make_form=make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze, summary_lines=_summary_lines),
)
