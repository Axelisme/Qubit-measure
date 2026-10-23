"""CPMG measurement, saved-source coherence fitting and operator guide."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Annotated, ClassVar, TypeAlias

from pydantic import BaseModel, Field
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    AnalyzeResultBase,
    ParamMeta,
    RunRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from .core import CPMG_Cfg, CPMG_Exp, CPMG_Result, CPMGAnalyzeOptions
from .form import make_form

CPMGRunResult: TypeAlias = RunRecord[CPMG_Cfg, CPMG_Result]


@dataclass
class CPMGAnalyzeParams:
    """Fringe fitting and optional reference coherence times in us."""

    fit_fringe: Annotated[bool, ParamMeta(label="Fit fringes")] = True
    t2r: Annotated[float | None, ParamMeta(label="Ramsey T2 reference (us)")] = None
    t1: Annotated[float | None, ParamMeta(label="T1 reference (us)")] = None


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
        description="T2 fit uncertainty in us, aligned with ns and t2s; null when unavailable."
    )


@dataclass
class CPMGAnalyzeResult(AnalyzeResultBase):
    """Pi-pulse counts and fitted T2/error in us; rejected rows are null."""

    ns: list[int]
    t2s: list[float | None]
    t2errs: list[float | None]

    def summary(self) -> CPMGSummary:
        """Return the named scalar summary consumed by recipes, in native units."""
        return CPMGSummary(
            ns=self.ns,
            t2s=self.t2s,
            t2errs=self.t2errs,
        )

    def to_summary_dict(self) -> dict[str, object]:
        return {**super().to_summary_dict(), **self.summary().model_dump()}


class CPMGAdapter(
    BaseAdapter[CPMG_Cfg, CPMGRunResult, CPMGAnalyzeResult, CPMGAnalyzeParams]
):
    """Acquire CPMG decay rows and fit their count dependence without writeback."""

    experiment_tag: ClassVar[str | None] = CPMG_Exp.persistence_tag()
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
        self, req: AnalyzeRequest[CPMGRunResult, CPMGAnalyzeParams], *, plots: Plots
    ) -> CPMGAnalyzeResult:
        """Fit the saved run and publish T2 per count, without changing calibration."""
        params = req.analyze_params
        result = CPMG_Exp().analyze(
            req.run_result,
            CPMGAnalyzeOptions(
                fit_fringe=params.fit_fringe, t2r=params.t2r, t1=params.t1
            ),
            plots=plots,
        )
        return CPMGAnalyzeResult(
            ns=result.ns.tolist(),
            t2s=[float(value) if isfinite(value) else None for value in result.t2s],
            t2errs=[
                float(error) if isfinite(value) else None
                for value, error in zip(result.t2s, result.t2errs, strict=True)
            ],
        )
