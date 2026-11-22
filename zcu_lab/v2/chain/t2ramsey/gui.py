from __future__ import annotations

from pathlib import Path

from pydantic import Field
from zcu_tools.analysis.readout_chain import CycleSums
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis, AnalyzeRequest
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.chain.t2ramsey.core import (
    ChainT2RamseyAnalysis,
    ChainT2RamseyCfg,
    ChainT2RamseyExp,
)
from zcu_lab.v2.chain.t2ramsey.form import make_form

from .._joint import JointResidualSummary


class ChainT2RamseySummary(JointResidualSummary):
    """Recipe JSON view of the analysis in native units; null is unavailable."""

    t2: float | None = Field(description="T2 Ramsey in us.")
    t2_err: float | None = Field(description="Wald error of T2 in us.")
    detuning: float | None = Field(description="Fringe frequency in MHz.")
    detuning_err: float | None = Field(description="Wald error of detuning in MHz.")
    tf: float | None = Field(description="Third-state decay time in us.")
    tf_err: float | None = Field(description="Wald error of tf in us.")
    reliability: float | None = Field(description="Diagnostic R, no threshold.")
    first_read_t2: float | None = Field(
        description="T2 in us of the first-read twotone fit, the starting value."
    )
    first_read_detuning: float | None = Field(
        description="Fringe MHz of the first-read twotone fit, the starting value."
    )


def _summary_lines(source: RunRecord[ChainT2RamseyCfg, CycleSums]) -> tuple[str, ...]:
    cfg = source.cfg
    return () if cfg is None else (f"Reps×Rounds {cfg.reps}×{cfg.rounds}",)


def _analyze(
    req: AnalyzeRequest[RunRecord[ChainT2RamseyCfg, CycleSums], None],
    *,
    plots: Plots,
) -> ChainT2RamseyAnalysis:
    return ChainT2RamseyExp().analyze(req.run_result, None, plots=plots)


def _run(
    cfg: ChainT2RamseyCfg, *, context: RunContext
) -> RunRecord[ChainT2RamseyCfg, CycleSums]:
    """Acquire the running sums and keep the experiment configuration with them."""
    return RunRecord(cfg=cfg, result=ChainT2RamseyExp().run(cfg, context=context))


# No writeback: the chain T2 is reported beside the first-read value; twotone/t2ramsey calibrates.
DEFINITION = Adapter(
    spec=ChainT2RamseyExp.AXES_SPEC,
    make_form=make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze, summary_lines=_summary_lines),
)
