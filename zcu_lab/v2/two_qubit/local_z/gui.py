"""Measure adapter for a compound, source-bound local-Z correction pair."""

from collections.abc import Sequence
from pathlib import Path
from typing import TypeAlias

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

from .core import LocalZAnalysis, LocalZCfg, LocalZExp, LocalZOptions, LocalZResult
from .form import make_form

LocalZRun: TypeAlias = RunRecord[LocalZCfg, LocalZResult]


def _analyze(req: AnalyzeRequest[LocalZRun, None], *, plots: Plots) -> LocalZAnalysis:
    """Fit saved counts, using only the fixed capture for display names."""
    parameters = req.parameters
    options = None
    if parameters is not None:
        options = LocalZOptions(
            qubit1_label=parameters.resolve_path("$qubit1"),
            qubit2_label=parameters.resolve_path("$qubit2"),
        )
    return LocalZExp().analyze(req.run_result, options, plots=plots)


def _get_writeback_items(
    req: WritebackRequest[LocalZRun, LocalZAnalysis],
) -> Sequence[WritebackItem]:
    """Propose one component-named pair only when both target fits are valid.

    Use only the fixed Run/Load capture for names and the coupler target.
    Raise ValueError without a capture. Apply does not alter the CZ pulses
    or their constructive relative phase.
    """
    parameters = req.ctx.parameters
    if parameters is None:
        raise ValueError("Local-Z writeback requires a Run/Load capture")
    first, second = req.analyze_result.fits
    if first.correction_deg is None or second.correction_deg is None:
        return []
    return [
        ParameterWriteback(
            target_name="$coupler.ext.cz_phase",
            description="B00 thermal linear-model logical corrections (deg); source Qubit1/Qubit2 identities, correction = −slope",
            proposed_value={
                "qubit1": parameters.resolve_path("$qubit1"),
                "qubit2": parameters.resolve_path("$qubit2"),
                "correction_qubit1_deg": first.correction_deg,
                "correction_qubit2_deg": second.correction_deg,
            },
        )
    ]


def run(cfg: LocalZCfg, *, context: RunContext) -> RunRecord[LocalZCfg, LocalZResult]:
    """Run the experiment and retain its actual configuration."""
    return RunRecord(cfg=cfg, result=LocalZExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=LocalZExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze, writeback=_get_writeback_items),
)
