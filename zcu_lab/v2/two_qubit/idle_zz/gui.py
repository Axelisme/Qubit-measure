"""Measure attachment for conditional Ramsey; acquisition never writes calibration."""

from dataclasses import dataclass
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

from ._analysis import IdleZZAnalysis
from .core import IdleZZCfg, IdleZZExp, IdleZZResult
from .form import make_form

IdleZZRun: TypeAlias = RunRecord[IdleZZCfg, IdleZZResult]


@dataclass(frozen=True)
class IdleZZAnalyzeResult:
    """JSON-safe analysis carrier; analysis retains branch fits and nullable ζ."""

    analysis: IdleZZAnalysis

    def to_summary_dict(self) -> dict[str, object]:
        """Expose signed/magnitude coordinates, fit quality and reasons without NaN.

        None means unavailable, not zero. Interpretation remains conditional on
        the saved operator detune-side assumption and thermal mixture limitations.
        """
        rows: list[dict[str, object]] = []
        for fit in self.analysis.branches:
            rows.append(
                {
                    "dc_index": fit.dc_index,
                    "spectator": "I" if fit.branch == 0 else "Xπ",
                    "complete": fit.complete,
                    "acquired_shots": fit.acquired_shots,
                    "magnitude_mhz": fit.magnitude_mhz,
                    "signed_frequency_mhz": fit.signed_mhz,
                    "frequency_error_mhz": fit.error_mhz,
                    "sign_resolution_mhz": fit.resolution_mhz,
                    "reason": fit.reason,
                    "fit_quality": None
                    if fit.fit_quality is None
                    else fit.fit_quality.to_summary_dict(),
                }
            )
        return {
            "branches": rows,
            "zeta_mhz": list(self.analysis.zeta_mhz),
            "zeta_error_mhz": list(self.analysis.zeta_error_mhz),
            "zeta_reasons": list(self.analysis.zeta_reasons),
            "interpretation": "f=sign(saved detune)*fringe magnitude; ζ=f_Xπ−f_I assumes both branches stay on the operator-selected detune side. Two nominal branches under a single-frequency model assumption; thermal mixture is not excluded. Model applicability is judged by the operator from fit_quality/plots, without an automatic statistical gate. Not pure ZZ; no heralding or writeback.",
        }


def _analyze(
    req: AnalyzeRequest[IdleZZRun, None],
    *,
    plots: Plots,
) -> IdleZZAnalyzeResult:
    """Fit saved cfg/raw only; current aliases, Form and library are ignored."""
    return IdleZZAnalyzeResult(IdleZZExp().analyze(req.run_result, None, plots=plots))


def run(cfg: IdleZZCfg, *, context: RunContext) -> RunRecord[IdleZZCfg, IdleZZResult]:
    """Run the experiment and retain its actual configuration."""
    return RunRecord(cfg=cfg, result=IdleZZExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=IdleZZExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze),
)
