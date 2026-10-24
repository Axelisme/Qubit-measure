"""Measure attachment for thermal four-state joint readout."""

from collections.abc import Sequence
from pathlib import Path
from typing import TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.plotting.plots import Plots

from .core import (
    JointReadoutAnalysis,
    JointReadoutCfg,
    JointReadoutExp,
    JointReadoutResult,
)
from .form import make_form

JointReadoutRun: TypeAlias = RunRecord[JointReadoutCfg, JointReadoutResult]


def _analyze(
    req: AnalyzeRequest[JointReadoutRun, None], *, plots: Plots
) -> JointReadoutAnalysis:
    """Classify complete branches using saved cfg; never current parameters."""
    return JointReadoutExp().analyze(req.run_result, None, plots=plots)


def _get_writeback_items(
    req: WritebackRequest[JointReadoutRun, JointReadoutAnalysis],
) -> Sequence[WritebackItem]:
    """Return no proposals: this thermal matrix is a diagnostic, not a correction."""
    del req
    return []


def run(
    cfg: JointReadoutCfg, *, context: RunContext
) -> RunRecord[JointReadoutCfg, JointReadoutResult]:
    """Run the experiment and retain its actual configuration."""
    return RunRecord(cfg=cfg, result=JointReadoutExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=JointReadoutExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze, writeback=_get_writeback_items),
)
