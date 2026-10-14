from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
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
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.singleshot_shared import read_confusion_matrix
from zcu_lab.v2.singleshot.reset_check.core import (
    ResetCheckAnalyzeOptions,
    ResetCheckCfg,
    ResetCheckExp,
    ResetCheckResult,
)
from zcu_lab.v2.singleshot.reset_check.form import FORM

SsResetCheckRunResult = RunRecord[ResetCheckCfg, ResetCheckResult]


@dataclass
class SsResetCheckAnalyzeResult(AnalyzeResultBase):
    reset_mean_excited_population: float
    reset_max_excited_population: float
    reset_max_other_population: float
    worst_sample_gain: float
    analyzed_reset_points: int


class SsResetCheckAdapter(
    BaseAdapter[
        ResetCheckCfg, SsResetCheckRunResult, SsResetCheckAnalyzeResult, NoAnalyzeParams
    ]
):
    experiment_tag: ClassVar[str | None] = ResetCheckExp.persistence_tag()
    form = FORM
    exp_cls = ResetCheckExp
    ExpCfg_cls = ResetCheckCfg
    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Hardware gain and three-branch sweep: Rabi / Rabi + reset / Rabi + reset + "
            "Rabi. Both pulses share the swept gain. Saves classified G/E populations; "
            "live view and analysis show Ground / Excited / Other with color for state "
            "and solid / dashed / dotted lines for branch."
        ),
        expects_md=(
            "Open and Reset capture the current aliases, parameters and library. Relax "
            "delay is 5 * $qubit.t1; the gain sweep ends at 4 * $qubit.module.x180.gain. "
            "Classification seeds $qubit.ext.readout_cal.g_center/e_center/radius; "
            "missing calibration stays Invalid. Inline templates read $qubit.freq, "
            "$qubit.wiring.L01.ch and $resonator.freq/wiring.ch/wiring.ro_ch; readout "
            "trigger uses $resonator.ext.time_of_fly + 0.05 us. Missing required sources "
            "remain Invalid; no numeric fallback is supplied."
        ),
        expects_ml=(
            "Rabi uses $qubit.module.x180 or pi_pulse. Tested reset uses "
            "$qubit.module.reset or reset template. Upstream reset is optional; readout "
            "uses $qubit.module.readout or readout template. Sweep-overwritten pulse "
            "values remain editable and library keys stay selectable. Run uses this fixed "
            "capture."
        ),
        typical_writeback=(
            "No writeback. Reports reset-only mean and worst sampled excited population "
            "and maximum Other fraction. No raw-IQ readout refit or histogram confidence "
            "intervals."
        ),
        recommended=(
            "Reps are shots per gain per branch per round; rounds average repeated "
            "hardware sweeps and update the live plot. Branches are independent "
            "preparations. Other is not calibrated leakage; population estimates are not "
            "reset-channel fidelity."
        ),
    )

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> SsResetCheckRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        result = ResetCheckExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self,
        req: AnalyzeRequest[SsResetCheckRunResult, NoAnalyzeParams],
        *,
        plots: Plots,
    ) -> SsResetCheckAnalyzeResult:
        analysis = ResetCheckExp().analyze(
            req.run_result,
            ResetCheckAnalyzeOptions(
                confusion_matrix=read_confusion_matrix(req.parameters)
            ),
            plots=plots,
        )
        return SsResetCheckAnalyzeResult(
            analysis.reset_mean_excited_population,
            analysis.reset_max_excited_population,
            analysis.reset_max_other_population,
            analysis.worst_sample_gain,
            analysis.analyzed_reset_points,
        )

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_ss_reset_check_{time.strftime('%m%d')}"
