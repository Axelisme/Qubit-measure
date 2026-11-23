from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

from zcu_tools.analysis.readout_chain import ThermalChainSums
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis, AnalyzeRequest
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.twotone.ro_optimize._chain_common import ChainAnalysis
from zcu_lab.v2.twotone.ro_optimize.chain_freq.core import ChainFreqCfg, ChainFreqExp
from zcu_lab.v2.twotone.ro_optimize.chain_freq.form import make_form

ChainFreqRunResult: TypeAlias = RunRecord[ChainFreqCfg, ThermalChainSums]


def _analyze(
    req: AnalyzeRequest[ChainFreqRunResult, None], *, plots: Plots
) -> ChainAnalysis:
    return ChainFreqExp().analyze(req.run_result, None, plots=plots)


def _summary_lines(source: ChainFreqRunResult) -> tuple[str, ...]:
    cfg = source.cfg
    if cfg is None:
        return ()
    return (f"Reps × Rounds: {cfg.reps} × {cfg.rounds}",)


def _run(cfg: ChainFreqCfg, *, context: RunContext) -> ChainFreqRunResult:
    """Acquire the moment sums and keep the experiment configuration with them."""
    return RunRecord(cfg=cfg, result=ChainFreqExp().run(cfg, context=context))


# No writeback: choose the maximum score; QND and M remain diagnostics.
DEFINITION = Adapter(
    spec=ChainFreqExp.AXES_SPEC,
    make_form=make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze, summary_lines=_summary_lines),
)
