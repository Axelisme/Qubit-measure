"""Measure adapter for source-bound paired two-qubit RB."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
)
from zcu_tools.plotting.plots import Plots

from .analysis import RBAnalysis, RBOptions
from .core import RBCfg, RBExp, RBResult
from .form import make_form

RBRun: TypeAlias = RunRecord[RBCfg, RBResult]


@dataclass(frozen=True)
class RBAnalyzeResult:
    """Opaque saved-source analysis; numeric summary carries no writeback."""

    analysis: RBAnalysis
    """Joint00 diagnostics, optional d=4 estimates and explicit failure reason."""

    def to_summary_dict(self) -> dict[str, object]:
        """Return JSON-safe estimates/reason and whole-round availability counts."""
        return {
            "estimate": None
            if self.analysis.estimate is None
            else asdict(self.analysis.estimate),
            "fit_quality": None
            if self.analysis.fit_quality is None
            else [q.to_summary_dict() for q in self.analysis.fit_quality],
            "reason": self.analysis.reason,
            "acquired_rounds": self.analysis.acquired_rounds.sum(axis=3).tolist(),
            "interpretation": (
                "d=4 error=3/4*(1-p_interleaved/p_reference), not clipped. "
                "Same-shot joint00 on all thermal shots; no heralding, leakage "
                "classification or readout inversion. Whole paired-seed bootstrap "
                "is statistical only; SPAM, finite samples and nonexponential or "
                "gate-dependent/coherent noise can bias interpretation."
            ),
        }


def _analyze(req: AnalyzeRequest[RBRun, None], *, plots: Plots) -> RBAnalyzeResult:
    """Count joint00 and fit only the saved source, never the current Form."""
    return RBAnalyzeResult(RBExp().analyze(req.run_result, RBOptions(), plots=plots))


def run(cfg: RBCfg, *, context: RunContext) -> RunRecord[RBCfg, RBResult]:
    """Run the experiment and retain its actual configuration."""
    return RunRecord(cfg=cfg, result=RBExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=RBExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze),
)
