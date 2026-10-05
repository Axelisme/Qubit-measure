"""AllXY gate-error check adapter.

Runs the 21 standard AllXY gate pairs and reuses ``AllXY_Exp.analyze``, which
fits power and detuning errors; the summary reports them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    ParamMeta,
    RunRequest,
    SessionEnv,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    scaled_md,
)
from zcu_lab.v2.twotone.allxy.core import (
    AllXY_Exp,
    AllXY_Result,
    AllXYAnalyzeOptions,
    AllXYCfg,
)

AllXYRunResult: TypeAlias = RunRecord[AllXYCfg, AllXY_Result]


@dataclass
class AllXYAnalyzeParams:
    fit_ge: Annotated[bool, ParamMeta(label="Fit g/e levels")] = False


@dataclass
class AllXYAnalyzeResult(AnalyzeResultBase):
    power_param: float
    detune_param: float
    power_err: float
    detune_err: float


class AllXYAdapter(
    BaseAdapter[AllXYCfg, AllXYRunResult, AllXYAnalyzeResult, AllXYAnalyzeParams]
):
    exp_cls = AllXY_Exp
    ExpCfg_cls: ClassVar[Any] = AllXYCfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "AllXY: runs the 21 standard pairs of I, X90, Y90, X180 and Y180 "
            "gates before readout. A calibrated qubit gives a staircase of "
            "ground, equator and excited levels; amplitude and detuning errors "
            "bend it in distinct patterns. Y gates are the X pulses with phase "
            "+90 deg. Runs on real hardware. Run after amplitude Rabi and "
            "Ramsey have calibrated the pi and pi/2 pulses and the frequency."
        ),
        expects_md=(
            "Reads 't1' (us) to seed relax_delay as 5*t1 (fallback ~30 us). "
            "Pulse modules pull 'q_f' (~2000–6000 MHz) and 'qub_ch'; readout "
            "pulls 'r_f', 'res_ch' / 'ro_ch' and 'timeFly'."
        ),
        expects_ml=(
            "Needs an X90 pulse module (prefers the calibrated library pi/2 "
            "pulse 'pi2_amp' / 'pi2_len'), an X180 pulse module (prefers "
            "'pi_amp' / 'pi_len'), and a readout module (calibrated "
            "'readout_dpm' / 'readout_rf' / 'readout' / 'res_readout', else a "
            "blank pulse-readout). Optional reset references a library reset "
            "when present, else stays disabled. The identity gate is the X90 "
            "pulse at zero gain."
        ),
        typical_writeback=(
            "No writeback. The summary reports 'power_err' and 'detune_err' "
            "(mean state deviation over the 21 pairs, also in the figure "
            "title) and the fitted model parameters 'power_param' and "
            "'detune_param'. Fix power errors with amplitude Rabi or zig-zag "
            "and detuning errors with Ramsey."
        ),
        recommended=(
            "Leave 'Fit g/e levels' off to take the ground and excited levels "
            "from the data range; turn it on when the measured levels are "
            "noisy or do not reach both states. Use many reps and rounds: "
            "each point is a single gate pair."
        ),
    )

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder()
            .reset(optional=True)
            .pulse("X90_pulse", role_id="pi2_pulse", label="X90 Pulse")
            .pulse("X180_pulse", role_id="pi_pulse", label="X180 Pulse")
            .readout()
            .relax_delay(scaled_md("t1", factor=5.0, fallback_value=30.5))
            .reps(1000)
            .rounds(100)
            .build()
        )

    def run(
        self, req: RunRequest, raw_cfg: dict[str, object], *, context: RunContext
    ) -> AllXYRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        return RunRecord(cfg=cfg, result=AllXY_Exp().run(cfg, context=context))

    def analyze(
        self,
        req: AnalyzeRequest[AllXYRunResult, AllXYAnalyzeParams],
        *,
        plots: Plots,
    ) -> AllXYAnalyzeResult:
        analysis = AllXY_Exp().analyze(
            req.run_result,
            AllXYAnalyzeOptions(fit_ge=req.analyze_params.fit_ge),
            plots=plots,
        )
        return AllXYAnalyzeResult(
            power_param=analysis.power_param,
            detune_param=analysis.detune_param,
            power_err=analysis.power_err,
            detune_err=analysis.detune_err,
        )

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_allxy_{time.strftime('%m%d')}"
