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

import time
from dataclasses import dataclass
from typing import ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    NoAnalyzeParams,
    RunRequest,
    SessionEnv,
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


@dataclass
class SsT1ToneSweepAnalyzeResult(AnalyzeResultBase):
    pass


class _SsT1ToneSweepBase(
    BaseAdapter[
        T1WithToneSweepCfg,
        SsT1ToneSweepRunResult,
        SsT1ToneSweepAnalyzeResult,
        NoAnalyzeParams,
    ]
):
    """Shared body for the two T1-with-tone-sweep adapters.

    Subclasses declare the single outer sweep axis (gain XOR freq) so the lowered
    cfg sets only one of them and the domain ``_resolve_outer_sweep`` is satisfied.
    """

    exp_cls = T1WithToneSweepExp

    # Subclasses supply the Form, analysis axis label and filename token.
    outer_label: ClassVar[str]
    filename_token: ClassVar[str]

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
        req: AnalyzeRequest[SsT1ToneSweepRunResult, NoAnalyzeParams],
        *,
        plots: Plots,
    ) -> SsT1ToneSweepAnalyzeResult:
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
        return SsT1ToneSweepAnalyzeResult()

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_ss_t1_tone_sweep_{self.filename_token}_{time.strftime('%m%d')}"


class SsT1ToneSweepGainAdapter(_SsT1ToneSweepBase):
    @staticmethod
    def make_form(ctx: FormContext) -> Form[T1WithToneSweepCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form_gain(ctx)

    experiment_tag: ClassVar[str | None] = T1WithToneSweepExp.persistence_tag()
    outer_label: ClassVar[str] = "Probe gain (a.u.)"
    filename_token: ClassVar[str] = "gain"

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Single-shot T1-with-tone, probe-gain sweep: at each probe gain it sweeps the "
            "wait-plus-probe length from both |g> and |e> initial states (Branch), "
            "classifies each shot in-program against the |g>/|e> centres, fits the dual "
            "transition rates per gain, and renders the 3×3 rate landscape over (gain, "
            "time). Runs on real hardware."
        ),
        expects_md=(
            "Open and Reset capture the current aliases, parameters and library. Relax "
            "delay and length stop are 5 * $qubit.t1. Frequency form sweeps "
            "$resonator.freq +/- 1.5 * $resonator.kappa; gain form sweeps 0 to 1. Each "
            "form exposes only one outer axis. Classification seeds "
            "$qubit.ext.readout_cal.g_center/e_center/radius; missing calibration stays "
            "Invalid. Inline templates read $qubit.freq, $qubit.wiring.L01.ch and "
            "$resonator.freq/wiring.ch/wiring.ro_ch; readout trigger uses "
            "$resonator.ext.time_of_fly + 0.05 us. Missing required sources remain "
            "Invalid; no numeric fallback is supplied."
        ),
        expects_ml=(
            "Pi pulse uses $qubit.module.x180 or pi_pulse. Probe starts from res_probe. "
            "Reset is optional; readout uses $qubit.module.readout or readout template. "
            "Sweep-overwritten pulse values remain editable and library keys stay "
            "selectable. Run uses this fixed capture."
        ),
        typical_writeback=(
            "No writeback — the rate landscape is read off the grid by eye."
        ),
        recommended=(
            "Set calibration cfg directly or seed it with 'singleshot/ge'. Keep the gain "
            "sweep coarse (the inner length sweep multiplies the run time). Provide "
            "'$qubit.ext.ac_stark_coeff' for a photon-number x-axis. "
            "'uniform=True' (default) uses a "
            "linear length sweep; set False to cluster points along the expected "
            "exponential decay while preserving the configured window and count."
        ),
    )


class SsT1ToneSweepFreqAdapter(_SsT1ToneSweepBase):
    @staticmethod
    def make_form(ctx: FormContext) -> Form[T1WithToneSweepCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form_freq(ctx)

    experiment_tag: ClassVar[str | None] = T1WithToneSweepExp.persistence_tag()
    outer_label: ClassVar[str] = "Probe frequency (MHz)"
    filename_token: ClassVar[str] = "freq"

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Single-shot T1-with-tone, probe-frequency sweep: at each probe frequency it "
            "sweeps the wait-plus-probe length from both |g> and |e> initial states "
            "(Branch), classifies each shot in-program against the |g>/|e> centres, fits "
            "the dual transition rates per frequency, and renders the 3×3 rate landscape "
            "over (freq, time). Runs on real hardware."
        ),
        expects_md=(
            "Open and Reset capture the current aliases, parameters and library. Relax "
            "delay and length stop are 5 * $qubit.t1. Frequency form sweeps "
            "$resonator.freq +/- 1.5 * $resonator.kappa; gain form sweeps 0 to 1. Each "
            "form exposes only one outer axis. Classification seeds "
            "$qubit.ext.readout_cal.g_center/e_center/radius; missing calibration stays "
            "Invalid. Inline templates read $qubit.freq, $qubit.wiring.L01.ch and "
            "$resonator.freq/wiring.ch/wiring.ro_ch; readout trigger uses "
            "$resonator.ext.time_of_fly + 0.05 us. Missing required sources remain "
            "Invalid; no numeric fallback is supplied."
        ),
        expects_ml=(
            "Pi pulse uses $qubit.module.x180 or pi_pulse. Probe starts from res_probe. "
            "Reset is optional; readout uses $qubit.module.readout or readout template. "
            "Sweep-overwritten pulse values remain editable and library keys stay "
            "selectable. Run uses this fixed capture."
        ),
        typical_writeback=(
            "No writeback — the rate landscape is read off the grid by eye."
        ),
        recommended=(
            "Set calibration cfg directly or seed it with 'singleshot/ge'. Keep the "
            "frequency sweep coarse (the inner length sweep multiplies the run time). "
            "'uniform=True' (default) uses a linear length sweep; set False to cluster "
            "points along the expected exponential decay while preserving the configured "
            "window and count."
        ),
    )
