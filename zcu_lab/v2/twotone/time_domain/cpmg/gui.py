"""CPMG measurement, saved-source coherence fitting and operator guide."""

from pathlib import Path
from typing import TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.utils import make_sweep
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis, AnalyzeRequest
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from .core import CPMG_Cfg, CPMG_Exp, CPMG_Result, CPMGAnalysis, CPMGAnalyzeOptions
from .form import CPMGInput, make_form

CPMGRunResult: TypeAlias = RunRecord[CPMG_Cfg, CPMG_Result]


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
        count=inp.count,
        delay=make_sweep(
            start=inp.min_length_us, stop=inp.max_length_us, expts=inp.length_points
        ),
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
