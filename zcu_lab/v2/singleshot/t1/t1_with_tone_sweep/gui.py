"""Single-shot T1-with-tone-sweep adapters (split by outer sweep axis).

The domain ``T1WithToneSweepExp`` is a 2D experiment: it always sweeps the
probe ``length`` plus exactly one *outer* axis (gain XOR freq), enforced by its
``_resolve_outer_sweep`` (it raises unless exactly one of gain/freq is set).
``T1WithToneSweepSweepCfg`` makes both gain and freq ``Optional``.

Rather than expose both optional sweeps in one form (which would let a user set
zero or two and hit the domain's run-time error — least-surprise violation),
this module splits into two adapters that each expose ``length`` + exactly one
outer sweep:

- ``SsT1ToneSweepGainAdapter`` → length + gain (registry 'singleshot/t1_tone_sweep_gain')
- ``SsT1ToneSweepFreqAdapter`` → length + freq (registry 'singleshot/t1_tone_sweep_freq')

Each lowers a cfg with only its one outer sweep set, so the domain's
``_resolve_outer_sweep`` is satisfied by construction. The shared run / analyze /
writeback / guide logic lives in ``_SsT1ToneSweepBase``; the subclasses differ
only in which outer sweep they declare + its default.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    RunRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
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


class _SsT1ToneSweepBase(BaseAdapter[T1WithToneSweepCfg, SsT1ToneSweepRunResult]):
    """Shared body for the two T1-with-tone-sweep adapters.

    Subclasses declare the single outer sweep axis (gain XOR freq) so the lowered
    cfg sets only one of them and the domain ``_resolve_outer_sweep`` is satisfied.
    """

    exp_cls = T1WithToneSweepExp

    # Subclasses supply the Form, analysis axis label and filename token.
    outer_label: ClassVar[str]

    def run(
        self,
        req: RunRequest,
        cfg: T1WithToneSweepCfg,
        *,
        context: RunContext,
    ) -> SsT1ToneSweepRunResult:
        del req
        return RunRecord(cfg, T1WithToneSweepExp().run(cfg, context=context))

    def analyze(
        self,
        req: AnalyzeRequest[SsT1ToneSweepRunResult, None],
        *,
        plots: Plots,
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
                ac_coeff=ac_coeff, confusion_matrix=confusion, xlabel=self.outer_label
            ),
            plots=plots,
        )
        return


class SsT1ToneSweepGainAdapter(_SsT1ToneSweepBase):
    @staticmethod
    def make_form(ctx: FormContext) -> Form[T1WithToneSweepCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form_gain(ctx)

    outer_label: ClassVar[str] = "Probe gain (a.u.)"

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.gain.md")


class SsT1ToneSweepFreqAdapter(_SsT1ToneSweepBase):
    @staticmethod
    def make_form(ctx: FormContext) -> Form[T1WithToneSweepCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form_freq(ctx)

    outer_label: ClassVar[str] = "Probe frequency (MHz)"

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.freq.md")
