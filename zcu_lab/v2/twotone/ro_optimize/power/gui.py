from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal, TypeAlias

from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    ParameterWriteback,
    ParamMeta,
    RunRequest,
    SessionEnv,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.writeback_helpers import readout_dpm_writeback_items
from zcu_lab.v2.twotone.ro_optimize.power.core import (
    PowerAnalyzeOptions,
    PowerCfg,
    PowerExp,
    PowerResult,
)
from zcu_lab.v2.twotone.ro_optimize.power.form import FORM

RoOptPowerRunResult: TypeAlias = RunRecord[PowerCfg, PowerResult]


@dataclass
class RoOptPowerAnalyzeParams:
    smooth_method: Annotated[
        Literal["wavelet", "gaussian"], ParamMeta(label="Smooth method")
    ] = "wavelet"
    smooth: Annotated[float, ParamMeta(label="Smooth strength", decimals=2)] = 1.0
    penalty_ratio: Annotated[
        float, ParamMeta(label="Power penalty ratio", decimals=2)
    ] = 0.5


@dataclass
class RoOptPowerAnalyzeResult(AnalyzeResultBase):
    best_gain: float


class RoOptPowerAdapter(
    BaseAdapter[
        PowerCfg,
        RoOptPowerRunResult,
        RoOptPowerAnalyzeResult,
        RoOptPowerAnalyzeParams,
    ]
):
    experiment_tag: ClassVar[str | None] = PowerExp.persistence_tag()
    exp_cls = PowerExp
    ExpCfg_cls = PowerCfg
    form = FORM

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Readout power optimization: with the qubit toggled between g and "
            "e by a pi pulse, sweeps the readout gain and measures the g/e "
            "signal-to-noise ratio (SNR), to pick the readout power that best "
            "distinguishes the states. Runs on real hardware. A readout-tuning "
            "step, typically after the readout frequency is set."
        ),
        expects_md=(
            "Seeds 101 gain points from 0.001 to 0.2. Relax delay reads 5.0 * $qubit.t1. "
            "Open and Reset evaluate starting expressions once into direct input using "
            "captured sources. Inline qubit pulses read $qubit.freq and "
            "$qubit.wiring.L01.ch. Inline readout reads $resonator.freq, "
            "$resonator.wiring.ch, $resonator.wiring.ro_ch and $resonator.ext.time_of_fly "
            "+ 0.05 us for trigger timing. Missing or invalid required sources leave the "
            "affected fields empty and Invalid; supply calibration or direct values "
            "before Run. No numeric calibration fallback is supplied."
        ),
        expects_ml=(
            "The qubit pulse uses $qubit.module.x180 as a captured library link, "
            "otherwise a fresh pi_pulse template with a const waveform. "
            "$qubit.module.readout selects a captured library link, otherwise a fresh "
            "inline readout template with a const waveform. Its required parameter "
            "sources must still be present. Optional $qubit.module.reset has no fallback: "
            "an unset/dangling slot or unbound alias leaves it disabled and Valid."
        ),
        typical_writeback=(
            "Proposes the SNR-maximizing readout gain into parameter context "
            "$qubit.ext.best_ro_gain (a.u.). When a cfg snapshot with pulse readout is "
            "available and $qubit.ext.best_ro_freq / $qubit.ext.best_ro_gain / "
            "$qubit.ext.best_ro_length are known from this result plus parameter context, "
            "also proposes $qubit.readout.readout_dpm."
        ),
        recommended=(
            "Analysis denoises the SNR curve before picking the peak; wavelet "
            "smoothing is the default. The 'power penalty ratio' down-weights "
            "high gains (SNR × exp(-gain × ratio)), biasing the choice toward "
            "lower power to limit measurement-induced effects; ~0.5 is a "
            "sensible default."
        ),
    )

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> RoOptPowerRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        return RunRecord(cfg, PowerExp().run(cfg, context=context))

    def analyze(
        self,
        req: AnalyzeRequest[RoOptPowerRunResult, RoOptPowerAnalyzeParams],
        *,
        plots: Plots,
    ) -> RoOptPowerAnalyzeResult:
        params = req.analyze_params
        options = PowerAnalyzeOptions(
            penalty_ratio=params.penalty_ratio,
            smooth=params.smooth,
            smooth_method=params.smooth_method,
        )
        result = PowerExp().analyze(req.run_result, options, plots=plots)
        return RoOptPowerAnalyzeResult(
            best_gain=result.best_gain,
        )

    def get_writeback_items(
        self, req: WritebackRequest[RoOptPowerRunResult, RoOptPowerAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        items: list[WritebackItem] = [
            ParameterWriteback(
                target_name="$qubit.ext.best_ro_gain",
                description="Optimal readout gain (a.u.)",
                proposed_value=result.best_gain,
            ),
        ]
        items.extend(
            readout_dpm_writeback_items(
                req.ctx,
                req.run_result.cfg,
                proposed={"best_ro_gain": result.best_gain},
            )
        )
        return items

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_ro_opt_gain_{time.strftime('%m%d')}"
