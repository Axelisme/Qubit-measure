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

from zcu_lab.v2.twotone.reset.flux.length.core import (
    LengthAnalysis,
    LengthAnalyzeOptions,
    LengthCfg,
    LengthExp,
    LengthResult,
)
from zcu_lab.v2.twotone.reset.flux.length.form import make_form


def _run(cfg: LengthCfg, *, context: RunContext) -> RunRecord[LengthCfg, LengthResult]:
    """Retain the fixed gain, pi calibration and readout with the four IQs."""
    return RunRecord(cfg=cfg, result=LengthExp().run(cfg, context=context))


def _analyze(
    req: AnalyzeRequest[RunRecord[LengthCfg, LengthResult], LengthAnalyzeOptions],
    *,
    plots: Plots,
) -> LengthAnalysis:
    return LengthExp().analyze(req.run_result, req.analyze_params, plots=plots)


def _writeback(
    req: WritebackRequest[RunRecord[LengthCfg, LengthResult], LengthAnalysis],
) -> Sequence[WritebackItem]:
    """Propose only a qualifying length; never Apply it or reuse an old value."""
    length = req.analyze_result.selected_length
    if length is None:
        return ()
    return (
        ParameterWriteback(
            target_name="$qubit.ext.flux_reset_length",
            description="Flux reset length (us), shortest sustained p0≈0 residual ≤ epsilon",
            proposed_value=length,
        ),
    )


DEFINITION = Adapter(
    spec=LengthExp.AXES_SPEC,
    make_form=make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_analyze,
        options=lambda source, env: LengthAnalyzeOptions(),
        writeback=_writeback,
    ),
)
