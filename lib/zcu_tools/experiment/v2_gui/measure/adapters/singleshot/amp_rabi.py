from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Literal

from matplotlib.figure import Figure

from zcu_tools.experiment.v2.singleshot.amp_rabi import (
    AmpRabiCfg,
    AmpRabiExp,
    AmpRabiFit,
    AmpRabiResult,
)
from zcu_tools.experiment.v2_gui.measure.adapters._support import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    ModuleInit,
    SweepDefault,
    scaled_md,
)
from zcu_tools.experiment.v2_gui.measure.adapters.base import BaseAdapter
from zcu_tools.gui.app.measure.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    ParamMeta,
    SessionEnv,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.cfg import EvalValue, ScalarSpec

from ._rabi import rabi_calibration_writeback


@dataclass
class SsAmpRabiAnalyzeParams:
    initial_state: Annotated[
        Literal["ground", "excited"], ParamMeta(label="Initial State")
    ] = "ground"


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
    BaseAdapter[
        AmpRabiCfg, AmpRabiResult, SsAmpRabiAnalyzeResult, SsAmpRabiAnalyzeParams
    ]
):
    exp_cls = AmpRabiExp
    ExpCfg_cls: ClassVar[Any] = AmpRabiCfg
    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior="Hardware gain sweep preserving every raw IQ shot. Live view classifies Ground / Excited / Other; analysis shares the Len Rabi joint IQ fit with no decay and fixed zero phase.",
        expects_md="Freezes g_center / e_center / radius from resolved cfg for live classification. Set calibration directly or seed defaults with singleshot/ge writeback; invalid calibration fails before hardware. Analysis jointly refits readout calibration without an external confusion matrix. Reads pi_gain for the sweep range.",
        expects_ml="Needs qub_pulse and readout; upstream reset is optional.",
        typical_writeback="Reports pi/pi2 gain, frequency and amplitude. A valid joint fit proposes g_center, e_center, ge_radius and confusion_matrix using the same calibration gate as Len Rabi.",
        recommended="Sweep a full Rabi period. Initial State describes the predominant state before the drive, at zero gain, even if the first gain is nonzero. Reps are shots per gain per round; rounds concatenate shots and update live populations. Population-only files cannot supply the raw IQ required for joint fitting.",
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
            .reps(1000)
            .rounds(1)
            .build()
        )

    def analyze(
        self, req: AnalyzeRequest[AmpRabiResult, SsAmpRabiAnalyzeParams]
    ) -> SsAmpRabiAnalyzeResult:
        fit, figure = AmpRabiExp().analyze(
            req.run_result, initial_state=req.analyze_params.initial_state
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

    def get_writeback_items(
        self, req: WritebackRequest[AmpRabiResult, SsAmpRabiAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        return rabi_calibration_writeback(req.analyze_result.fit_result.joint_fit)

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_ss_amp_rabi_{time.strftime('%m%d')}"
