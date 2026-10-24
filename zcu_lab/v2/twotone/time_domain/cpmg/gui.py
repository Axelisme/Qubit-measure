"""CPMG measurement, saved-source coherence fitting and operator guide."""

from pathlib import Path
from typing import TypeAlias

from pydantic import BaseModel, Field
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis, AnalyzeRequest
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from .core import CPMG_Cfg, CPMG_Exp, CPMG_Result, CPMGAnalysis, CPMGAnalyzeOptions
from .form import CPMGInput, make_form

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


def _CPMG_analyze(
    req: AnalyzeRequest[CPMGRunResult, CPMGAnalyzeOptions], *, plots: Plots
) -> CPMGAnalysis:
    """Fit the saved run and publish T2 per count, without changing calibration."""
    return CPMG_Exp().analyze(
        req.run_result,
        req.analyze_params,
        plots=plots,
    )


def _CPMG_make_form(ctx: FormContext) -> Form[CPMGInput]:
    """Build shared time-window controls and captured module selections."""
    return make_form(ctx)


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
        fn=_CPMG_analyze, options=lambda source, env: CPMGAnalyzeOptions()
    ),
)
