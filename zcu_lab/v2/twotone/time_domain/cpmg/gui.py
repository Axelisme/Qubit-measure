"""CPMG measurement, saved-source coherence fitting and operator guide."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Annotated, TypeAlias

from pydantic import BaseModel, Field
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    AnalyzeResultBase,
    ParamMeta,
)
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from .core import CPMG_Cfg, CPMG_Exp, CPMG_Result, CPMGAnalyzeOptions
from .form import CPMGInput, make_form

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


def _CPMG_make_form(ctx: FormContext) -> Form[CPMGInput]:
    """Build shared time-window controls and captured module selections."""
    return make_form(ctx)


def _CPMG_analyze(
    req: AnalyzeRequest[CPMGRunResult, CPMGAnalyzeParams], *, plots: Plots
) -> CPMGAnalyzeResult:
    """Fit the saved run and publish T2 per count, without changing calibration."""
    params = req.analyze_params
    result = CPMG_Exp().analyze(
        req.run_result,
        CPMGAnalyzeOptions(fit_fringe=params.fit_fringe, t2r=params.t2r, t1=params.t1),
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


def _run(inp: CPMGInput, *, context: RunContext) -> RunRecord[CPMG_Cfg, CPMG_Result]:
    """Apply the shared time window to every pi-pulse count and record it."""
    cfg = CPMG_Cfg(
        reps=inp.reps,
        rounds=inp.rounds,
        relax_delay=inp.relax_delay,
        reset=inp.reset,
        pi2_pulse=inp.pi2_pulse,
        pi_pulse=inp.pi_pulse,
        readout=inp.readout,
        times=inp.times,
        length_range=(inp.min_length_us, inp.max_length_us),
        length_expts=inp.length_points,
        detune_ratio=inp.detune_ratio,
        earlystop_snr=inp.earlystop_snr,
    )
    return RunRecord(cfg=cfg, result=CPMG_Exp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=CPMG_Exp.AXES_SPEC,
    make_form=_CPMG_make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_CPMG_analyze, options=lambda source, env: CPMGAnalyzeParams()
    ),
)
