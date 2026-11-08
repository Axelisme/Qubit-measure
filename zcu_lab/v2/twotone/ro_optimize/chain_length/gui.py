from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

from zcu_tools.analysis.readout_chain import ReadoutSums
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis, AnalyzeRequest
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.twotone.ro_optimize.chain_common import ChainAnalysis
from zcu_lab.v2.twotone.ro_optimize.chain_length.core import (
    ChainLengthCfg,
    ChainLengthExp,
)
from zcu_lab.v2.twotone.ro_optimize.chain_length.form import make_form

ChainLengthRunResult: TypeAlias = RunRecord[ChainLengthCfg, ReadoutSums]


def _analyze(
    req: AnalyzeRequest[ChainLengthRunResult, None], *, plots: Plots
) -> ChainAnalysis:
    return ChainLengthExp().analyze(req.run_result, None, plots=plots)


def _run(cfg: ChainLengthCfg, *, context: RunContext) -> ChainLengthRunResult:
    """Acquire the moment sums and keep the experiment configuration with them."""
    return RunRecord(cfg=cfg, result=ChainLengthExp().run(cfg, context=context))


# No writeback: the operator chooses the window from the metric and d/sigma.
DEFINITION = Adapter(
    spec=ChainLengthExp.AXES_SPEC,
    make_form=make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze),
)
