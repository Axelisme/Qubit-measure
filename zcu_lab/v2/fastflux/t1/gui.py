"""Measure GUI for gain-resolved relaxation during a fast-flux pulse."""

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
)
from zcu_tools.plotting.plots import Plots

from .core import T1Analysis, T1Cfg, T1Exp, T1Result
from .form import make_form

T1RunResult: TypeAlias = RunRecord[T1Cfg, T1Result]


@dataclass(frozen=True)
class T1AnalyzeResult:
    """Gain-resolved decay estimates; rejected fits are null in the summary."""

    gains: list[float]
    analysis: T1Analysis

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "gains": self.gains,
            "t1_us": [
                float(value) if np.isfinite(value) else None
                for value in self.analysis.t1s
            ],
            "t1_err_us": [
                float(error) if np.isfinite(value) and np.isfinite(error) else None
                for value, error in zip(
                    self.analysis.t1s, self.analysis.t1errs, strict=True
                )
            ],
        }


def _analyze(
    req: AnalyzeRequest[T1RunResult, None], *, plots: Plots
) -> T1AnalyzeResult:
    source = req.run_result
    return T1AnalyzeResult(
        gains=source.result.gains.tolist(),
        analysis=T1Exp().analyze(source, None, plots=plots),
    )


DEFINITION = Adapter(
    spec=T1Exp.AXES_SPEC,
    make_form=make_form,
    run=T1Exp().run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze),
)
