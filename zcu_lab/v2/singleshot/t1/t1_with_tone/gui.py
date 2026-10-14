from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar, TypeAlias

from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    NoAnalyzeParams,
    ParameterWriteback,
    RunRequest,
    SessionEnv,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.singleshot_shared import (
    read_confusion_matrix,
)
from zcu_lab.v2.singleshot.t1.t1_with_tone.core import (
    T1WithToneAnalyzeOptions,
    T1WithToneCfg,
    T1WithToneExp,
    T1WithToneResult,
)
from zcu_lab.v2.singleshot.t1.t1_with_tone.form import FORM

# The numeric t1 is written to the t1_with_tone parameter context key.
SsT1ToneRunResult: TypeAlias = RunRecord[T1WithToneCfg, T1WithToneResult]


@dataclass
class SsT1ToneAnalyzeResult(AnalyzeResultBase):
    t1: float
    t1_b: float


class SsT1ToneAdapter(
    BaseAdapter[
        T1WithToneCfg,
        SsT1ToneRunResult,
        SsT1ToneAnalyzeResult,
        NoAnalyzeParams,
    ]
):
    experiment_tag: ClassVar[str | None] = T1WithToneExp.persistence_tag()
    form = FORM
    exp_cls = T1WithToneExp
    ExpCfg_cls = T1WithToneCfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Single-shot T1-with-tone: applies a π pulse and a simultaneous probe tone "
            "during the wait, sweeps the wait-plus-probe length, and classifies each shot "
            "in-program against the |g>/|e> IQ-cluster centres. Plots the ground / "
            "excited / other populations versus delay time with dual-transition-rate fits "
            "for T1 and T1_b. The fitted T1 value is written back to 't1_with_tone' in "
            "the parameter context. Runs on real hardware."
        ),
        expects_md=(
            "Open and Reset capture the current aliases, parameters and library. Relax "
            "delay is 5 * $qubit.t1. Length stop is 5 * $qubit.ext.t1_with_tone; there is "
            "no fallback to t1. Classification seeds "
            "$qubit.ext.readout_cal.g_center/e_center/radius; missing calibration stays "
            "Invalid. Inline templates read $qubit.freq, $qubit.wiring.L01.ch and "
            "$resonator.freq/wiring.ch/wiring.ro_ch; readout trigger uses "
            "$resonator.ext.time_of_fly + 0.05 us. Missing required sources remain "
            "Invalid; no numeric fallback is supplied."
        ),
        expects_ml=(
            "Pi pulse uses $qubit.module.x180 or pi_pulse. Probe starts from res_probe. "
            "Optional init/reset use $qubit.module.x180/reset. Readout uses "
            "$qubit.module.readout or readout template. Sweep-overwritten pulse values "
            "remain editable and library keys stay selectable. Run uses this fixed "
            "capture."
        ),
        typical_writeback=(
            "Proposes the fitted T1 relaxation time under the probe tone into parameter "
            "context 't1_with_tone' (us)."
        ),
        recommended=(
            "Set calibration cfg directly or seed it with 'singleshot/ge'. Use "
            "'uniform=False' (default) to cluster points along the expected exponential "
            "decay while preserving the configured window and point count; use "
            "'uniform=True' for a linear sweep. The probe-tone gain and frequency are set "
            "inside the probe_pulse module."
        ),
    )

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> SsT1ToneRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        return RunRecord(cfg, T1WithToneExp().run(cfg, context=context))

    def analyze(
        self, req: AnalyzeRequest[SsT1ToneRunResult, NoAnalyzeParams], *, plots: Plots
    ) -> SsT1ToneAnalyzeResult:
        # ``confusion_matrix`` is the GE 3×3 readout-correction matrix from md.
        confusion = read_confusion_matrix(req.parameters)
        result = T1WithToneExp().analyze(
            req.run_result,
            T1WithToneAnalyzeOptions(confusion_matrix=confusion),
            plots=plots,
        )
        return SsT1ToneAnalyzeResult(t1=result.t1, t1_b=result.t1_b)

    def get_writeback_items(
        self,
        req: WritebackRequest[SsT1ToneRunResult, SsT1ToneAnalyzeResult],
    ) -> Sequence[WritebackItem]:
        # Key ``t1_with_tone`` per single_qubit.md:3079.
        return [
            ParameterWriteback(
                target_name="t1_with_tone",
                description="T1 with probe tone (us)",
                proposed_value=req.analyze_result.t1,
            ),
        ]

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_ss_t1_tone_{time.strftime('%m%d')}"
