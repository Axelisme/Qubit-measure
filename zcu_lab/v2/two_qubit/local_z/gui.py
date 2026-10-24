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

from .analysis import LocalZAnalysis
from .core import LocalZCfg, LocalZExp, LocalZResult
from .form import make_form

LocalZRun: TypeAlias = RunRecord[LocalZCfg, LocalZResult]


def _analyze(req: AnalyzeRequest[LocalZRun, None], *, plots: Plots) -> LocalZAnalysis:
    """Fit only saved cfg/raw, never current Form, aliases or library."""
    return LocalZExp().analyze(req.run_result, None, plots=plots)


def _get_writeback_items(
    req: WritebackRequest[LocalZRun, LocalZAnalysis],
) -> Sequence[WritebackItem]:
    """Propose one component-named pair only when both target fits are valid.

    Saved cfg and Run/Load parameter capture are required. Raise ValueError
    if its F1/F2/coupler identities differ from the saved source, including
    swapped roles; never silently reuse a pair. Preview/selected Apply is
    manual. No pulse, constructive relative phase or other parameter changes.
    """
    cfg, parameters = req.run_result.cfg, req.ctx.parameters
    if cfg is None or parameters is None:
        return []
    for role, name in (
        ("f1", cfg.roles.f1),
        ("f2", cfg.roles.f2),
        ("coupler", cfg.roles.coupler),
    ):
        if parameters.resolve_path(f"${role}") != name:
            raise ValueError(
                "Local-Z source F1/F2/coupler identities differ from writeback capture; Reset and remeasure for changed roles"
            )
    first, second = req.analyze_result.fits
    if first.correction_deg is None or second.correction_deg is None:
        return []
    return [
        ParameterWriteback(
            target_name="$coupler.ext.cz_phase",
            description="B00 thermal linear-model logical corrections (deg); source F1/F2 identities, correction = −slope",
            proposed_value={
                "f1": cfg.roles.f1,
                "f2": cfg.roles.f2,
                "correction_f1_deg": first.correction_deg,
                "correction_f2_deg": second.correction_deg,
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
