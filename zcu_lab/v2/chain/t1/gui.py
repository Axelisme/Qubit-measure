from __future__ import annotations

from pathlib import Path

from pydantic import Field
from zcu_tools.analysis.readout_chain import CycleSums
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis, AnalyzeRequest
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.chain.t1.core import ChainT1Analysis, ChainT1Cfg, ChainT1Exp
from zcu_lab.v2.chain.t1.form import make_form

from .._joint import JointResidualSummary


class ChainT1Summary(JointResidualSummary):
    """Recipe JSON view of the analysis in native units; null is unavailable."""

    t1: float | None = Field(
        description="T1 in us, the eigen time with the larger amplitude."
    )
    t1_err: float | None = Field(description="Wald error of T1 in us.")
    secondary_time: float | None = Field(
        description="The other eigen time in us; null when infinite."
    )
    secondary_time_err: float | None = Field(
        description="Wald error of the secondary time in us."
    )


def _summary_lines(source: RunRecord[ChainT1Cfg, CycleSums]) -> tuple[str, ...]:
    cfg = source.cfg
    return () if cfg is None else (f"Reps×Rounds {cfg.reps}×{cfg.rounds}",)


def _analyze(
    req: AnalyzeRequest[RunRecord[ChainT1Cfg, CycleSums], None],
    *,
    plots: Plots,
) -> ChainT1Analysis:
    return ChainT1Exp().analyze(req.run_result, None, plots=plots)


def _run(cfg: ChainT1Cfg, *, context: RunContext) -> RunRecord[ChainT1Cfg, CycleSums]:
    """Acquire the running sums and keep the experiment configuration with them."""
    return RunRecord(cfg=cfg, result=ChainT1Exp().run(cfg, context=context))


# No writeback: twotone/time_domain/t1 calibrates $qubit.t1.
DEFINITION = Adapter(
    spec=ChainT1Exp.AXES_SPEC,
    make_form=make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze, summary_lines=_summary_lines),
)
