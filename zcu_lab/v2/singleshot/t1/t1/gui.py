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

from zcu_lab.v2._support.measure.singleshot_shared import read_confusion_matrix
from zcu_lab.v2.singleshot.t1.t1.core import T1AnalyzeOptions, T1Cfg, T1Exp, T1Result
from zcu_lab.v2.singleshot.t1.t1.form import make_form

# Transition-rate analysis publishes figures without numeric writeback.
SsT1RunResult: TypeAlias = RunRecord[T1Cfg, T1Result]


@dataclass
class SsT1AnalyzeResult(AnalyzeResultBase):
    pass


class SsT1Adapter(
    BaseAdapter[T1Cfg, SsT1RunResult, SsT1AnalyzeResult, NoAnalyzeParams]
):
    experiment_tag: ClassVar[str | None] = T1Exp.persistence_tag()

    @staticmethod
    def make_form(ctx: FormContext) -> Form[T1Cfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    exp_cls = T1Exp

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Single-shot T1: applies a π pulse, waits for a variable delay, reads out, "
            "and classifies each shot in-program against the |g>/|e> IQ-cluster centres. "
            "Repeats from both |g> and |e> initial states (Branch). Plots the ground / "
            "excited / other populations versus delay time with dual-transition-rate fits "
            "for T1 and T1_b. Runs on real hardware."
        ),
        expects_md=(
            "Open and Reset capture the current aliases, parameters and library. Relax "
            "delay and length stop are 5 * $qubit.t1. Classification seeds "
            "$qubit.ext.readout_cal.g_center/e_center/radius; missing calibration stays "
            "Invalid. Inline templates read $qubit.freq, $qubit.wiring.L01.ch and "
            "$resonator.freq/wiring.ch/wiring.ro_ch; readout trigger uses "
            "$resonator.ext.time_of_fly + 0.05 us. Missing required sources remain "
            "Invalid; no numeric fallback is supplied."
        ),
        expects_ml=(
            "Pi pulse uses $qubit.module.x180 or pi_pulse. Reset is optional; readout "
            "uses $qubit.module.readout or readout template. Sweep-overwritten pulse "
            "values remain editable and library keys stay selectable. Run uses this fixed "
            "capture."
        ),
        typical_writeback=(
            "No writeback — T1 is shown in the plot title only. Use 'singleshot/t1_tone' "
            "if you need the T1 scalar in the parameter context."
        ),
        recommended=(
            "Set calibration cfg directly or seed it with 'singleshot/ge'. A delay sweep "
            "reaching ~5*T1 lets the decay flatten. Missing t1 leaves the length stop and "
            "relax delay Invalid. Set 'uniform=True' to sweep linearly; leave False to "
            "cluster more points along the expected exponential decay while preserving "
            "the configured start/stop window and point count."
        ),
    )

    def run(
        self,
        req: RunRequest,
        cfg: T1Cfg,
        *,
        context: RunContext,
    ) -> SsT1RunResult:
        del req
        return RunRecord(cfg, T1Exp().run(cfg, context=context))

    def analyze(
        self, req: AnalyzeRequest[SsT1RunResult, NoAnalyzeParams], *, plots: Plots
    ) -> SsT1AnalyzeResult:
        # ``confusion_matrix`` is the GE 3×3 readout-correction matrix from md.
        # ``skip`` is not exposed as a user knob — users can re-run with a shorter
        # sweep instead.
        confusion = read_confusion_matrix(req.parameters)
        T1Exp().analyze(
            req.run_result, T1AnalyzeOptions(confusion_matrix=confusion), plots=plots
        )
        return SsT1AnalyzeResult()

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_ss_t1_{time.strftime('%m%d')}"
