from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    ParameterWriteback,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.twotone.reset.flux.gain.core import (
    GainAnalysis,
    GainCfg,
    GainExp,
    GainResult,
)
from zcu_lab.v2.twotone.reset.flux.gain.form import make_form


def _run(cfg: GainCfg, *, context: RunContext) -> RunRecord[GainCfg, GainResult]:
    """Retain the fixed duration, pi calibration and readout with the four IQs."""
    return RunRecord(cfg=cfg, result=GainExp().run(cfg, context=context))


def _analyze(
    req: AnalyzeRequest[RunRecord[GainCfg, GainResult], None], *, plots: Plots
) -> GainAnalysis:
    return GainExp().analyze(req.run_result, None, plots=plots)


def _writeback(
    req: WritebackRequest[RunRecord[GainCfg, GainResult], GainAnalysis],
) -> Sequence[WritebackItem]:
    """Propose the selected gain only; no finite candidate produces no proposal."""
    gain = req.analyze_result.selected_gain
    if gain is None:
        return ()
    return (
        ParameterWriteback(
            target_name="$qubit.ext.flux_reset_gain",
            description="Flux reset gain (a.u.), minimum p0≈0 residual estimate",
            proposed_value=gain,
        ),
    )


DEFINITION = Adapter(
    spec=GainExp.AXES_SPEC,
    make_form=make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze, writeback=_writeback),
)
