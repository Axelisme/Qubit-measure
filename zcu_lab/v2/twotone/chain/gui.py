from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from zcu_tools.analysis.readout_chain import SweepSums
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis, AnalyzeRequest
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.twotone.chain.core import ChainAnalysis, ChainCfg, ChainExp
from zcu_lab.v2.twotone.chain.form import make_form
from zcu_lab.v2.twotone.freq.analysis import FreqAnalyzeOptions


class TwotoneChainSummary(BaseModel):
    """Recipe JSON inputs in native units; null means an unavailable estimate."""

    freq: float | None = Field(
        description="Sigma-peak centre in MHz; null if unavailable."
    )
    freq_err: float | None = Field(
        description="Centre error in MHz; null if unavailable."
    )
    fwhm: float | None = Field(
        description="Sigma-peak FWHM in MHz, an approximate line width; null if unavailable."
    )
    fwhm_err: float | None = Field(
        description="FWHM error in MHz; null if unavailable."
    )
    level: float | None = Field(description="Dashed 5% false-mark level in sigma.")


def _analyze(
    req: AnalyzeRequest[RunRecord[ChainCfg, SweepSums], FreqAnalyzeOptions],
    *,
    plots: Plots,
) -> ChainAnalysis:
    return ChainExp().analyze(req.run_result, req.analyze_params, plots=plots)


def _run(cfg: ChainCfg, *, context: RunContext) -> RunRecord[ChainCfg, SweepSums]:
    """Acquire the running sums and keep the experiment configuration with them."""
    return RunRecord(cfg=cfg, result=ChainExp().run(cfg, context=context))


# No writeback: the sigma-peak width only approximates the qubit line width.
DEFINITION = Adapter(
    spec=ChainExp.AXES_SPEC,
    make_form=make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_analyze,
        options=lambda source, env: FreqAnalyzeOptions(),
    ),
)
