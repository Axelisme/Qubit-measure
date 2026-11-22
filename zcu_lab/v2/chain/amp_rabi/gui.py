from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from zcu_tools.analysis.readout_chain import CycleCheck, CycleSums
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis, AnalyzeRequest
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.chain.amp_rabi.core import (
    ChainAmpRabiAnalysis,
    ChainAmpRabiCfg,
    ChainAmpRabiExp,
)
from zcu_lab.v2.chain.amp_rabi.form import make_form


class ChainAmpRabiSummary(BaseModel):
    """Recipe JSON view of the analysis in native units; null is unavailable."""

    pi_amp: float | None = Field(description="Pi gain 1/(2f) in a.u.")
    pi_amp_err: float | None = Field(description="Pi gain error in a.u.")
    f: float | None = Field(description="Rabi frequency in cycles per gain unit.")
    f_err: float | None = Field(description="Wald error of f.")
    reliability: float | None = Field(description="Diagnostic R, no threshold.")
    within_point: CycleCheck = Field(description="Within-point check J, df, p.")
    sharing: CycleCheck = Field(description="Sharing check J, df; p is null.")
    curve: CycleCheck = Field(description="Curve check J, df, p.")
    first_read_pi_amp: float | None = Field(
        description="Pi gain of the first-read twotone fit, the starting value."
    )


def _analyze(
    req: AnalyzeRequest[RunRecord[ChainAmpRabiCfg, CycleSums], None],
    *,
    plots: Plots,
) -> ChainAmpRabiAnalysis:
    return ChainAmpRabiExp().analyze(req.run_result, None, plots=plots)


def _run(
    cfg: ChainAmpRabiCfg, *, context: RunContext
) -> RunRecord[ChainAmpRabiCfg, CycleSums]:
    """Acquire the running sums and keep the experiment configuration with them."""
    return RunRecord(cfg=cfg, result=ChainAmpRabiExp().run(cfg, context=context))


# No writeback: the chain π gain is reported beside the first-read value; twotone/rabi/amp_rabi calibrates.
DEFINITION = Adapter(
    spec=ChainAmpRabiExp.AXES_SPEC,
    make_form=make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze),
)
