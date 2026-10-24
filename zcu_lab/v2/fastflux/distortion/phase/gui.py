"""Measure and display flux-transient phase tomography in the measure GUI."""

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    AnalyzeResultBase,
)
from zcu_tools.plotting.plots import Plots

from .core import PhaseCfg, PhaseExp, PhaseResult
from .form import make_form

PhaseRunResult: TypeAlias = RunRecord[PhaseCfg, PhaseResult]


@dataclass
class PhaseAnalyzeResult(AnalyzeResultBase):
    """Unwrapped phase versus pair midpoint; arrays stay out of scalar summary."""

    lengths_us: NDArray[np.float64]
    phases_deg: NDArray[np.float64]


def _analyze(
    req: AnalyzeRequest[PhaseRunResult, None],
    *,
    plots: Plots,
) -> PhaseAnalyzeResult:
    """Plot tomography and unwrapped phase using only the source run timing."""
    analysis = PhaseExp().analyze(req.run_result, None, plots=plots)
    return PhaseAnalyzeResult(
        lengths_us=analysis.lengths_us, phases_deg=analysis.phases_deg
    )


DEFINITION = Adapter(
    spec=PhaseExp.AXES_SPEC,
    make_form=make_form,
    run=PhaseExp().run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze),
)
