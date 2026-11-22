from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from zcu_tools.analysis.readout_chain import CycleCheck, CycleSums
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis, AnalyzeRequest
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.chain.t2echo.core import (
    ChainT2EchoAnalysis,
    ChainT2EchoCfg,
    ChainT2EchoExp,
)
from zcu_lab.v2.chain.t2echo.form import make_form


class ChainT2EchoSummary(BaseModel):
    """Recipe JSON view of the analysis in native units; null is unavailable."""

    t2: float | None = Field(description="T2 echo in us.")
    t2_err: float | None = Field(description="Wald error of T2 echo in us.")
    tf: float | None = Field(description="Third-state decay time in us.")
    tf_err: float | None = Field(description="Wald error of tf in us.")
    within_point: CycleCheck = Field(description="Within-point check J, df, p.")
    sharing: CycleCheck = Field(description="Sharing check J, df; p is null.")
    curve: CycleCheck = Field(description="Curve check J, df, p.")
    first_read_t2: float | None = Field(
        description="T2 echo in us of the first-read twotone fit, the starting value."
    )


def _summary_lines(source: RunRecord[ChainT2EchoCfg, CycleSums]) -> tuple[str, ...]:
    cfg = source.cfg
    return () if cfg is None else (f"Reps×Rounds {cfg.reps}×{cfg.rounds}",)


def _analyze(
    req: AnalyzeRequest[RunRecord[ChainT2EchoCfg, CycleSums], None],
    *,
    plots: Plots,
) -> ChainT2EchoAnalysis:
    return ChainT2EchoExp().analyze(req.run_result, None, plots=plots)


def _run(
    cfg: ChainT2EchoCfg, *, context: RunContext
) -> RunRecord[ChainT2EchoCfg, CycleSums]:
    """Acquire the running sums and keep the experiment configuration with them."""
    return RunRecord(cfg=cfg, result=ChainT2EchoExp().run(cfg, context=context))


# No writeback: the chain T2 echo is reported beside the first-read value; twotone/t2echo calibrates.
DEFINITION = Adapter(
    spec=ChainT2EchoExp.AXES_SPEC,
    make_form=make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze, summary_lines=_summary_lines),
)
