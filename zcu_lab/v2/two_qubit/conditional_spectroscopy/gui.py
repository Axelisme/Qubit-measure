"""Measure adapter for four thermal branches of two-line spectroscopy."""

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
)
from zcu_tools.plotting.plots import Plots

from .analysis import SpectroscopyAnalysis
from .core import BRANCH_LABELS, SpectroscopyCfg, SpectroscopyExp, SpectroscopyResult
from .form import make_form

SpectroscopyRun: TypeAlias = RunRecord[SpectroscopyCfg, SpectroscopyResult]


@dataclass(frozen=True)
class SpectroscopyAnalyzeResult:
    """JSON-safe completion summary; diagnostic spectra remain named figures."""

    analysis: SpectroscopyAnalysis

    def to_summary_dict(self) -> dict[str, object]:
        """Report all DC/branch shot counts, never label incomplete means as valid."""
        rows: list[dict[str, object]] = []
        for d in range(self.analysis.complete.shape[0]):
            for b, label in enumerate(BRANCH_LABELS):
                complete = bool(self.analysis.complete[d, b])
                acquired = int(self.analysis.acquired_shots[d, b])
                rows.append(
                    {
                        "dc_index": d,
                        "branch": f"B{label}",
                        "status": "complete"
                        if complete
                        else "incomplete"
                        if acquired
                        else "not_started",
                        "reason": None
                        if complete
                        else "Expected frequencies/rounds not fully acquired",
                        "acquired_shots": acquired,
                        "analyzed_shots": int(self.analysis.analyzed_shots[d, b]),
                    }
                )
        return {
            "branches": rows,
            "interpretation": "Thermal nominal preparations; all shots and full tones retained. Mean I/Q are diagnostic proxies, not bare-level assignments or leakage populations. No heralding, peak fit, Hamiltonian fit or calibration writeback.",
        }


def _analyze(
    req: AnalyzeRequest[SpectroscopyRun, None], *, plots: Plots
) -> SpectroscopyAnalyzeResult:
    """Use only source cfg/raw; never read current parameters or propose peaks."""
    return SpectroscopyAnalyzeResult(
        SpectroscopyExp().analyze(req.run_result, None, plots=plots)
    )


DEFINITION = Adapter(
    spec=SpectroscopyExp.AXES_SPEC,
    make_form=make_form,
    run=SpectroscopyExp().run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze),
)
