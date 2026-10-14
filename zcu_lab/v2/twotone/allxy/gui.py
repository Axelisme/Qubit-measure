"""AllXY gate-error check adapter.

Runs the 21 standard AllXY gate pairs and reuses ``AllXY_Exp.analyze``, which
fits power and detuning errors; the summary reports them.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, TypeAlias

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
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.twotone.allxy.core import (
    AllXY_Exp,
    AllXY_Result,
    AllXYAnalyzeOptions,
    AllXYCfg,
)

from .form import FORM

AllXYRunResult: TypeAlias = RunRecord[AllXYCfg, AllXY_Result]


@dataclass
class AllXYAnalyzeParams:
    fit_ge: Annotated[bool, ParamMeta(label="Fit g/e levels")] = False


@dataclass
class AllXYAnalyzeResult(AnalyzeResultBase):
    amplitude_error: float
    detune_param: float
    power_err: float
    detune_err: float
    residual_rms: float


class AllXYAdapter(
    BaseAdapter[AllXYCfg, AllXYRunResult, AllXYAnalyzeResult, AllXYAnalyzeParams]
):
    form = FORM
    experiment_tag: ClassVar[str | None] = AllXY_Exp.persistence_tag()
    exp_cls = AllXY_Exp
    ExpCfg_cls: ClassVar[Any] = AllXYCfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "AllXY: runs the 21 standard pairs of I, X90, Y90, X180 and Y180 gates "
            "before readout. A calibrated qubit gives a staircase of ground, equator "
            "and excited levels; amplitude and detuning errors bend it in distinct "
            "patterns. Y gates are the X pulses with phase +90 deg. Runs on real "
            "hardware. Run after amplitude Rabi and Ramsey have calibrated the pi and "
            "pi/2 pulses and the frequency."
        ),
        expects_md=(
            "AllXY uses the captured gate slots. Relax delay is 5 * qubit.t1 us. "
            "Missing calibration leaves the affected fields empty and Invalid; no "
            "numeric fallback."
        ),
        expects_ml=(
            "Uses captured qubit.module.readout with a readout template fallback, and "
            "qubit.module.reset with no fallback (disabled when absent). Drive slots "
            "use qubit.module.x180/x90 with pi_pulse/pi2_pulse template fallbacks. "
            "Unset or dangling slots use the named template, not a legacy blank. Key "
            "choices remain available. Inline templates read qubit.freq, "
            "qubit.wiring.L01.ch, resonator.freq, resonator.wiring.ch/ro_ch and "
            "resonator.ext.time_of_fly + 0.05 us; missing required calibration remains "
            "Invalid."
        ),
        typical_writeback=(
            "No writeback. The summary reports 'amplitude_error', the relative drive "
            "amplitude error (+0.02 means 2% too much rotation), and 'detune_param', "
            "the dimensionless detuning parameter of the fit model. 'power_err' and "
            "'detune_err' are the mean Bloch-z deviation (ground +1, excited -1) each "
            "error causes over the 21 pairs; they are not gate infidelities. "
            "'residual_rms' is the fit residual in the same units; a residual "
            "comparable to those deviations means the model does not fit, for example "
            "because T1/T2 decay matters. Fix power errors with amplitude Rabi or "
            "zig-zag and detuning errors with Ramsey."
        ),
        recommended=(
            "Leave 'Fit g/e levels' off to take the ground and excited levels from the "
            "data range; turn it on when the measured levels are noisy or do not reach "
            "both states. Use many reps and rounds: each point is a single gate pair."
        ),
    )

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
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
            amplitude_error=analysis.amplitude_error,
            detune_param=analysis.detune_param,
            power_err=analysis.power_err,
            detune_err=analysis.detune_err,
            residual_rms=analysis.residual_rms,
        )

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_allxy_{time.strftime('%m%d')}"
