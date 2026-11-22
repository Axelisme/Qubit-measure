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
from zcu_tools.gui.session.parameters import DocumentSetCapture
from zcu_tools.plotting.plots import Plots
from zcu_tools.resources.entry import RoleResolutionError

from .core import LocalZAnalysis, LocalZCfg, LocalZExp, LocalZOptions, LocalZResult
from .form import make_form

LocalZRun: TypeAlias = RunRecord[LocalZCfg, LocalZResult]


def _captured_names(parameters: DocumentSetCapture | None) -> tuple[str, str] | None:
    if parameters is None:
        return None
    try:
        return parameters.resolve_path("$qubit1"), parameters.resolve_path("$qubit2")
    except RoleResolutionError as exc:
        if exc.reason not in (
            "Unknown alias",
            "Alias is not bound",
            "Role is not bound",
        ):
            raise
        return None


def _analyze(req: AnalyzeRequest[LocalZRun, None], *, plots: Plots) -> LocalZAnalysis:
    """Fit saved counts; use position labels when the source has no qubit pair."""
    names = _captured_names(req.parameters)
    options = LocalZOptions(*names) if names is not None else None
    return LocalZExp().analyze(req.run_result, options, plots=plots)


def _get_writeback_items(
    req: WritebackRequest[LocalZRun, LocalZAnalysis],
) -> Sequence[WritebackItem]:
    """Propose one component-named pair only when both target fits are valid.

    Use only the fixed Run/Load capture for names and the coupler target.
    Raise ValueError without both qubit bindings in the capture. Apply does
    not alter the CZ pulses or their constructive relative phase.
    """
    names = _captured_names(req.ctx.parameters)
    if names is None:
        raise ValueError(
            "Local-Z writeback requires a Run/Load capture with $qubit1 and $qubit2 bindings"
        )
    qubit1, qubit2 = names
    first, second = req.analyze_result.fits
    if first.correction_deg is None or second.correction_deg is None:
        return []
    return [
        ParameterWriteback(
            # Keys: $coupler.ext.cz_phase.qubit1, $coupler.ext.cz_phase.qubit2,
            # $coupler.ext.cz_phase.correction_qubit1_deg,
            # $coupler.ext.cz_phase.correction_qubit2_deg.
            target_name="$coupler.ext.cz_phase",
            description="B00 thermal linear-model logical corrections (deg); source Qubit1/Qubit2 identities, correction = −slope",
            proposed_value={
                "qubit1": qubit1,
                "qubit2": qubit2,
                "correction_qubit1_deg": first.correction_deg,
                "correction_qubit2_deg": second.correction_deg,
            },
        )
    ]


def run(cfg: LocalZCfg, *, context: RunContext) -> RunRecord[LocalZCfg, LocalZResult]:
    """Run the experiment and retain its actual configuration."""
    return RunRecord(cfg=cfg, result=LocalZExp().run(cfg, context=context))


def _summary_lines(source: LocalZRun) -> tuple[str, ...]:
    cfg = source.cfg
    if cfg is None:
        return ()
    return (f"Reps×Rounds {cfg.reps}×{cfg.rounds}",)


DEFINITION = Adapter(
    spec=LocalZExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_analyze, writeback=_get_writeback_items, summary_lines=_summary_lines
    ),
)
