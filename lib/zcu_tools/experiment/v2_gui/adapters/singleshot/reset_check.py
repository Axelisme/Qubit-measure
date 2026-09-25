from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, ClassVar

from matplotlib.figure import Figure

from zcu_tools.experiment.v2.singleshot.reset_check import (
    ResetCheckCfg,
    ResetCheckExp,
    ResetCheckResult,
)
from zcu_tools.experiment.v2_gui.adapters._support import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    SweepDefault,
    scaled_md,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.gui.app.main.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    ExpContext,
    NoAnalyzeParams,
    RunRequest,
    require_soc_handles,
)
from zcu_tools.gui.app.main.adapter.lowering import schema_to_raw_dict
from zcu_tools.gui.cfg import CfgSchema

from ._shared import read_ge_centers


@dataclass
class SsResetCheckAnalyzeResult(AnalyzeResultBase):
    reset_mean_excited_population: float
    reset_max_excited_population: float
    reset_max_other_population: float
    worst_sample_gain: float
    analyzed_reset_points: int
    figure: Figure


class SsResetCheckAdapter(
    BaseAdapter[
        ResetCheckCfg, ResetCheckResult, SsResetCheckAnalyzeResult, NoAnalyzeParams
    ]
):
    exp_cls = ResetCheckExp
    ExpCfg_cls: ClassVar[Any] = ResetCheckCfg
    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior="Hardware gain and three-branch sweep: Rabi / Rabi + reset / Rabi + reset + Rabi. Both pulses share the swept gain. Saves classified G/E populations; live view and analysis show Ground / Excited / Other with color for state and solid / dashed / dotted lines for branch.",
        expects_md="Requires g_center / e_center / ge_radius from singleshot/ge. Optionally uses confusion_matrix for analysis correction. Reads pi_gain and t1 for defaults.",
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
            .reps(5000)
            .rounds(1)
            .build()
        )

    def run(self, req: RunRequest, schema: CfgSchema) -> ResetCheckResult:
        soc, soccfg = require_soc_handles(req)
        cfg = self.build_exp_cfg(schema_to_raw_dict(schema, req.md, req.ml), req)
        g_center, e_center, radius = read_ge_centers(req.md)
        return ResetCheckExp().run(soc, soccfg, cfg, g_center, e_center, radius)

    def analyze(
        self, req: AnalyzeRequest[ResetCheckResult, NoAnalyzeParams]
    ) -> SsResetCheckAnalyzeResult:
        analysis, figure = ResetCheckExp().analyze(
            req.run_result, confusion_matrix=req.md.get("confusion_matrix")
        )
        return SsResetCheckAnalyzeResult(
            analysis.reset_mean_excited_population,
            analysis.reset_max_excited_population,
            analysis.reset_max_other_population,
            analysis.worst_sample_gain,
            analysis.analyzed_reset_points,
            figure,
        )

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_ss_reset_check_{time.strftime('%m%d')}"
