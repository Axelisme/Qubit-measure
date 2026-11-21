"""GUI wiring for single-qubit T1 COV."""

from pathlib import Path

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    ParameterWriteback,
    WritebackRequest,
)
from zcu_tools.plotting.plots import Plots

from .core import (
    T1CovAnalysis,
    T1CovAnalyzeOptions,
    T1CovCfg,
    T1CovExp,
    T1CovResult,
)
from .form import make_form


def _run(cfg: T1CovCfg, *, context: RunContext) -> RunRecord[T1CovCfg, T1CovResult]:
    """Retain acquisition cfg with unclassified paired IQ moments."""
    return RunRecord(cfg=cfg, result=T1CovExp().run(cfg, context=context))


def _analyze(
    req: AnalyzeRequest[RunRecord[T1CovCfg, T1CovResult], T1CovAnalyzeOptions],
    *,
    plots: Plots,
) -> T1CovAnalysis:
    """Fit self-calibrated C and publish fit parameters and population diagnostics."""
    return T1CovExp().analyze(req.run_result, req.analyze_params, plots=plots)


def _writeback(
    req: WritebackRequest[RunRecord[T1CovCfg, T1CovResult], T1CovAnalysis],
) -> list[ParameterWriteback]:
    """Propose T1 in us for the captured qubit; applying remains explicit."""
    return [
        ParameterWriteback(
            target_name="$qubit.t1",
            description="T1 relaxation time from paired IQ moments (µs)",
            proposed_value=req.analyze_result.t1,
            stderr=req.analyze_result.t1_err,
        )
    ]


DEFINITION = Adapter(
    spec=T1CovExp.AXES_SPEC,
    make_form=make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_analyze,
        options=lambda source, env: T1CovAnalyzeOptions(),
        writeback=_writeback,
    ),
)
