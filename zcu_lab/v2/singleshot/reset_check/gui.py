from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, ClassVar

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_lab.v2.singleshot.reset_check.core import ResetCheckAnalyzeOptions
from zcu_lab.v2.singleshot.reset_check.core import ResetCheckCfg
from zcu_lab.v2.singleshot.reset_check.core import ResetCheckExp
from zcu_lab.v2.singleshot.reset_check.core import ResetCheckResult
from zcu_lab.v2._support.measure.schema_builder import MeasureCfgBuilder
from zcu_lab.v2._support.measure.schema_builder import MeasureCfgDefinition
from zcu_lab.v2._support.measure.seeds import SweepDefault
from zcu_lab.v2._support.measure.seeds import scaled_md
from zcu_tools.gui.app.measure.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    NoAnalyzeParams,
    RunRequest,
    SessionEnv,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.cfg import EvalValue, ScalarSpec
from zcu_tools.plotting.plots import Plots

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
    exp_cls = ResetCheckExp
    ExpCfg_cls: ClassVar[Any] = ResetCheckCfg
    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior="Hardware gain and three-branch sweep: Rabi / Rabi + reset / Rabi + reset + Rabi. Both pulses share the swept gain. Saves classified G/E populations; live view and analysis show Ground / Excited / Other with color for state and solid / dashed / dotted lines for branch.",
        expects_md="Freezes g_center / e_center / radius from resolved cfg. Set calibration directly or seed defaults with singleshot/ge writeback; invalid calibration fails before hardware. Optionally uses confusion_matrix for analysis correction. Reads pi_gain and t1 for defaults.",
        expects_ml="Needs rabi_pulse, tested_reset and readout. Optional upstream reset prepares the initial state.",
        typical_writeback="No writeback. Reports reset-only mean and worst sampled excited population and maximum Other fraction. No raw-IQ readout refit or histogram confidence intervals.",
        recommended="Reps are shots per gain per branch per round; rounds average repeated hardware sweeps and update the live plot. Branches are independent preparations. Other is not calibrated leakage; population estimates are not reset-channel fidelity.",
    )

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder()
            .reset(optional=True)
            .pulse("rabi_pulse", role_id="rabi_pulse", label="Rabi Pulse")
            .reset("tested_reset", role_id="reset", label="Tested Reset")
            .readout()
            .relax_delay(scaled_md("t1", factor=5.0, fallback_value=100.0))
            .sweep(
                "gain",
                label="Gain (a.u.)",
                default=SweepDefault(
                    start=0.0,
                    stop=scaled_md("pi_gain", factor=4.0, fallback_value=1.0),
                    expts=51,
                ),
            )
            .field(
                "g_center",
                spec=ScalarSpec("Ground center", complex),
                default=EvalValue("g_center"),
            )
            .field(
                "e_center",
                spec=ScalarSpec("Excited center", complex),
                default=EvalValue("e_center"),
            )
            .field(
                "radius",
                spec=ScalarSpec("Classification radius", float),
                default=EvalValue("ge_radius"),
            )
            .reps(5000)
            .rounds(1)
            .build()
        )

    def run(
        self, req: RunRequest, raw_cfg: dict[str, object], *, context: RunContext
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
            ResetCheckAnalyzeOptions(confusion_matrix=req.md.get("confusion_matrix")),
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
