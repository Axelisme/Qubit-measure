from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    ParamMeta,
    RunRequest,
    SessionEnv,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.singleshot_rabi import rabi_calibration_writeback
from zcu_lab.v2.singleshot.amp_rabi.core import (
    AmpRabiAnalyzeOptions,
    AmpRabiCfg,
    AmpRabiExp,
    AmpRabiFit,
    AmpRabiResult,
)
from zcu_lab.v2.singleshot.amp_rabi.form import FORM

SsAmpRabiRunResult = RunRecord[AmpRabiCfg, AmpRabiResult]


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


class SsAmpRabiAdapter(
    BaseAdapter[
        AmpRabiCfg, SsAmpRabiRunResult, SsAmpRabiAnalyzeResult, SsAmpRabiAnalyzeParams
    ]
):
    experiment_tag: ClassVar[str | None] = AmpRabiExp.persistence_tag()
    form = FORM
    exp_cls = AmpRabiExp
    ExpCfg_cls = AmpRabiCfg
    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Hardware gain sweep preserving every raw IQ shot. Live view classifies "
            "Ground / Excited / Other; analysis shares the Len Rabi joint IQ fit with no "
            "decay and fixed zero phase."
        ),
        expects_md=(
            "Open and Reset capture the current aliases, parameters and library. Relax "
            "delay starts at 50.5 us. Gain sweep starts at -0.3 and ends at 4 * "
            "$qubit.module.x180.gain. Inline qubit gain starts at 1.0. Classification "
            "seeds $qubit.ext.readout_cal.g_center/e_center/radius; missing calibration "
            "stays Invalid. Inline templates read $qubit.freq, $qubit.wiring.L01.ch and "
            "$resonator.freq/wiring.ch/wiring.ro_ch; readout trigger uses "
            "$resonator.ext.time_of_fly + 0.05 us. Missing required sources remain "
            "Invalid; no numeric fallback is supplied."
        ),
        expects_ml=(
            "Qubit pulse starts from qub_probe. Reset uses optional $qubit.module.reset; "
            "readout uses $qubit.module.readout or readout template. Sweep-overwritten "
            "pulse values remain editable and library keys stay selectable. Run uses this "
            "fixed capture."
        ),
        typical_writeback=(
            "Reports pi/pi2 gain, frequency and amplitude. A valid joint fit proposes "
            "g_center, e_center, ge_radius and confusion_matrix using the same "
            "calibration gate as Len Rabi."
        ),
        recommended=(
            "Sweep a full Rabi period. Initial State describes the predominant state "
            "before the drive, at zero gain, even if the first gain is nonzero. Reps are "
            "shots per gain per round; rounds concatenate shots and update live "
            "populations. Population-only files cannot supply the raw IQ required for "
            "joint fitting."
        ),
    )

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> SsAmpRabiRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        result = AmpRabiExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self,
        req: AnalyzeRequest[SsAmpRabiRunResult, SsAmpRabiAnalyzeParams],
        *,
        plots: Plots,
    ) -> SsAmpRabiAnalyzeResult:
        fit = AmpRabiExp().analyze(
            req.run_result,
            AmpRabiAnalyzeOptions(initial_state=req.analyze_params.initial_state),
            plots=plots,
        )
        return SsAmpRabiAnalyzeResult(
            fit.pi_gain,
            fit.pi_gain_error,
            fit.pi2_gain,
            fit.pi2_gain_error,
            fit.frequency,
            fit.amplitude,
            fit,
        )

    def get_writeback_items(
        self, req: WritebackRequest[SsAmpRabiRunResult, SsAmpRabiAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        return rabi_calibration_writeback(req.analyze_result.fit_result.joint_fit)

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_ss_amp_rabi_{time.strftime('%m%d')}"
