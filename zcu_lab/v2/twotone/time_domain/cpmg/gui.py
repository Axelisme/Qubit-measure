"""CPMG measurement, saved-source coherence fitting and operator guide."""

from pathlib import Path
from typing import ClassVar, TypeAlias

from pydantic import BaseModel, Field
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    RunRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from .core import CPMG_Cfg, CPMG_Exp, CPMG_Result, CPMGAnalysis, CPMGAnalyzeOptions
from .form import make_form

CPMGRunResult: TypeAlias = RunRecord[CPMG_Cfg, CPMG_Result]


class CPMGSummary(BaseModel):
    """Row-aligned pi-pulse counts and T2 estimates; null means unavailable.

    ns[i], t2s[i] and t2errs[i] describe the same CPMG fit row.
    """

    ns: list[int] = Field(
        description="Number of pi pulses per CPMG sequence, in fit-row order."
    )

    t2s: list[float | None] = Field(
        description="Fitted T2 in us, aligned with ns; null when unavailable."
    )

    t2errs: list[float | None] = Field(
        description="Native T2 fit uncertainty in us, aligned with ns and t2s; nonfinite values are null. A rejected T2 row may retain zero uncertainty."
    )


class CPMGAdapter(
    BaseAdapter[CPMG_Cfg, CPMGRunResult, CPMGAnalysis, CPMGAnalyzeOptions]
):
    """Acquire CPMG decay rows and fit their count dependence without writeback."""

    exp_cls = CPMG_Exp
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @staticmethod
    def make_form(ctx: FormContext) -> Form[CPMG_Cfg]:
        """Build shared time-window controls and captured module selections."""
        return make_form(ctx)

    def run(
        self, req: RunRequest, cfg: CPMG_Cfg, *, context: RunContext
    ) -> CPMGRunResult:
        """Acquire the accepted count/time grid and retain its exact cfg."""
        del req
        return RunRecord(cfg=cfg, result=CPMG_Exp().run(cfg, context=context))

    def analyze(
        self, req: AnalyzeRequest[CPMGRunResult, CPMGAnalyzeOptions], *, plots: Plots
    ) -> CPMGAnalysis:
        """Fit the saved run and publish T2 per count, without changing calibration."""
        return CPMG_Exp().analyze(
            req.run_result,
            req.analyze_params,
            plots=plots,
        )
