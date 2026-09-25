from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, ClassVar

from matplotlib.figure import Figure

from zcu_tools.experiment.v2.singleshot.amp_rabi import (
    AmpRabiCfg,
    AmpRabiExp,
    AmpRabiFit,
    AmpRabiResult,
)
from zcu_tools.experiment.v2_gui.adapters._support import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    ModuleInit,
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
class SsAmpRabiAnalyzeResult(AnalyzeResultBase):
    pi_gain: float
    pi_gain_error: float
    pi2_gain: float
    pi2_gain_error: float
    frequency: float
    amplitude: float
    fit_result: AmpRabiFit
    figure: Figure


class SsAmpRabiAdapter(
    BaseAdapter[AmpRabiCfg, AmpRabiResult, SsAmpRabiAnalyzeResult, NoAnalyzeParams]
):
    exp_cls = AmpRabiExp
    ExpCfg_cls: ClassVar[Any] = AmpRabiCfg
    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior="Hardware gain sweep with G/E classification in the acquisition layer. Saves populations only. Live view shows Ground / Excited / Other; analysis fits a nondecaying ground-population Rabi curve.",
        expects_md="Requires g_center / e_center / ge_radius from singleshot/ge. Optionally uses confusion_matrix for analysis correction. Reads pi_gain for the sweep range.",
        expects_ml="Needs qub_pulse and readout; upstream reset is optional.",
        typical_writeback="No writeback. Reports pi/pi2 gain, frequency and oscillation amplitude; IQ centers and readout calibration are not refitted.",
        recommended="Sweep a full Rabi period with at least five gains. Reps are shots per gain per round; rounds average repeated hardware sweeps and update the live plot. Other is not calibrated leakage.",
    )

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder()
            .reset(optional=True)
            .pulse(
                "qub_pulse",
                role_id="qub_probe",
                init=ModuleInit.INLINE,
                overrides={"gain": 1.0},
            )
            .readout()
            .relax_delay(50.5)
            .sweep(
                "gain",
                label="Gain (a.u.)",
                default=SweepDefault(
                    start=-0.3,
                    stop=scaled_md("pi_gain", factor=4.0, fallback_value=1.0),
                    expts=51,
                ),
            )
            .reps(1000)
            .rounds(1)
            .build()
        )

    def run(self, req: RunRequest, schema: CfgSchema) -> AmpRabiResult:
        soc, soccfg = require_soc_handles(req)
        cfg = self.build_exp_cfg(schema_to_raw_dict(schema, req.md, req.ml), req)
        g_center, e_center, radius = read_ge_centers(req.md)
        return AmpRabiExp().run(soc, soccfg, cfg, g_center, e_center, radius)

    def analyze(
        self, req: AnalyzeRequest[AmpRabiResult, NoAnalyzeParams]
    ) -> SsAmpRabiAnalyzeResult:
        fit, figure = AmpRabiExp().analyze(
            req.run_result, confusion_matrix=req.md.get("confusion_matrix")
        )
        return SsAmpRabiAnalyzeResult(
            fit.pi_gain,
            fit.pi_gain_error,
            fit.pi2_gain,
            fit.pi2_gain_error,
            fit.frequency,
            fit.amplitude,
            fit,
            figure,
        )

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_ss_amp_rabi_{time.strftime('%m%d')}"
