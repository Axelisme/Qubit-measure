from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from zcu_tools.analysis.readout_chain import CycleCheck, CycleSums
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis, AnalyzeRequest
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.chain.len_rabi.core import (
    ChainLenRabiAnalysis,
    ChainLenRabiCfg,
    ChainLenRabiExp,
)
from zcu_lab.v2.chain.len_rabi.form import make_form


class ChainLenRabiSummary(BaseModel):
    """Recipe JSON view of the analysis in native units; null is unavailable."""

    pi_len: float | None = Field(description="Pi length 1/(2f) in us.")
    pi_len_err: float | None = Field(description="Pi length error in us.")
    f: float | None = Field(description="Rabi frequency in MHz.")
    f_err: float | None = Field(description="Wald error of f in MHz.")
    reliability: float | None = Field(description="Diagnostic R, no threshold.")
    within_point: CycleCheck = Field(description="Within-point check J, df, p.")
    sharing: CycleCheck = Field(description="Sharing check J, df; p is null.")
    curve: CycleCheck = Field(description="Curve check J, df, p.")
    first_read_pi_len: float | None = Field(
        description="Pi length in us of the first-read twotone fit, the starting value."
    )


def _summary_lines(source: RunRecord[ChainLenRabiCfg, CycleSums]) -> tuple[str, ...]:
    cfg = source.cfg
    return () if cfg is None else (f"Reps×Rounds {cfg.reps}×{cfg.rounds}",)


def _analyze(
    req: AnalyzeRequest[RunRecord[ChainLenRabiCfg, CycleSums], None],
    *,
    plots: Plots,
) -> ChainLenRabiAnalysis:
    return ChainLenRabiExp().analyze(req.run_result, None, plots=plots)


def _run(
    cfg: ChainLenRabiCfg, *, context: RunContext
) -> RunRecord[ChainLenRabiCfg, CycleSums]:
    """Acquire the running sums and keep the experiment configuration with them."""
    return RunRecord(cfg=cfg, result=ChainLenRabiExp().run(cfg, context=context))


# No writeback: the chain π length is reported beside the first-read value; twotone/rabi/len_rabi calibrates.
DEFINITION = Adapter(
    spec=ChainLenRabiExp.AXES_SPEC,
    make_form=make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze, summary_lines=_summary_lines),
)
