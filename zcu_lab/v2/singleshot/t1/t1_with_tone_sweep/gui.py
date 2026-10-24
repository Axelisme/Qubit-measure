"""Single-shot T1 versus probe length and either gain or frequency.

GAIN_DEFINITION exposes length + gain at 'singleshot/t1_tone_sweep_gain'.
FREQ_DEFINITION exposes length + frequency at 'singleshot/t1_tone_sweep_freq'.
Each form supplies exactly one outer sweep, as required by the core run.
Analysis shares the captured correction inputs but labels its own outer axis.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
)
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.singleshot_shared import (
    read_analysis_float,
    read_confusion_matrix,
)
from zcu_lab.v2.singleshot.t1.t1_with_tone_sweep.core import (
    T1WithToneSweepAnalyzeOptions,
    T1WithToneSweepCfg,
    T1WithToneSweepExp,
    T1WithToneSweepResult,
)
from zcu_lab.v2.singleshot.t1.t1_with_tone_sweep.form import (
    make_form_freq,
    make_form_gain,
)

SsT1ToneSweepRunResult: TypeAlias = RunRecord[T1WithToneSweepCfg, T1WithToneSweepResult]


def _analyze(
    req: AnalyzeRequest[SsT1ToneSweepRunResult, None],
    *,
    plots: Plots,
    xlabel: str,
) -> None:
    # ``ac_coeff`` (= md 'ac_stark_coeff') rescales the outer x-axis to photon
    # number; ``confusion_matrix`` readout-corrects; both are md inputs, not
    # user knobs. ``xlabel`` labels the rate plots with this adapter's outer
    # axis.
    ac_coeff = read_analysis_float(req.parameters, "$qubit.ext.ac_stark_coeff")
    confusion = read_confusion_matrix(req.parameters)
    T1WithToneSweepExp().analyze(
        req.run_result,
        T1WithToneSweepAnalyzeOptions(
            ac_coeff=ac_coeff, confusion_matrix=confusion, xlabel=xlabel
        ),
        plots=plots,
    )
    return


def _analyze_gain(
    req: AnalyzeRequest[SsT1ToneSweepRunResult, None], *, plots: Plots
) -> None:
    _analyze(req, plots=plots, xlabel="Probe gain (a.u.)")


GAIN_DEFINITION = Adapter(
    spec=T1WithToneSweepExp.AXES_SPEC,
    make_form=make_form_gain,
    run=T1WithToneSweepExp().run,
    guide=Path(__file__).with_name("guide.gain.md"),
    analysis=Analysis(fn=_analyze_gain),
)


def _analyze_freq(
    req: AnalyzeRequest[SsT1ToneSweepRunResult, None], *, plots: Plots
) -> None:
    _analyze(req, plots=plots, xlabel="Probe frequency (MHz)")


FREQ_DEFINITION = Adapter(
    spec=T1WithToneSweepExp.AXES_SPEC,
    make_form=make_form_freq,
    run=T1WithToneSweepExp().run,
    guide=Path(__file__).with_name("guide.freq.md"),
    analysis=Analysis(fn=_analyze_freq),
)
