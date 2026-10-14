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
from zcu_lab.v2.twotone.ro_optimize.freq.core import (
    FreqAnalyzeOptions,
    FreqCfg,
    FreqExp,
    FreqResult,
)
from zcu_lab.v2.twotone.ro_optimize.freq.form import FORM

RoOptFreqRunResult: TypeAlias = RunRecord[FreqCfg, FreqResult]


@dataclass
class RoOptFreqAnalyzeParams:
    smooth_method: Annotated[
        Literal["wavelet", "gaussian"], ParamMeta(label="Smooth method")
    ] = "wavelet"
    smooth: Annotated[float, ParamMeta(label="Smooth strength", decimals=2)] = 2.0


@dataclass
class RoOptFreqAnalyzeResult(AnalyzeResultBase):
    best_freq: float


class RoOptFreqAdapter(
    BaseAdapter[
        FreqCfg,
        RoOptFreqRunResult,
        RoOptFreqAnalyzeResult,
        RoOptFreqAnalyzeParams,
    ]
):
    experiment_tag: ClassVar[str | None] = FreqExp.persistence_tag()
    exp_cls = FreqExp
    ExpCfg_cls = FreqCfg
    form = FORM

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Readout frequency optimization: with the qubit prepared in g and "
            "e (a pi pulse toggles it), sweeps the readout frequency and "
            "measures the g/e signal-to-noise ratio (SNR), so you can pick the "
            "readout frequency that best distinguishes the two states. Runs on "
            "real hardware. One step of readout tuning; usually run after the "
            "qubit and a pi pulse are calibrated."
        ),
        expects_md=(
            "Seeds 301 frequency points at $resonator.freq +/- $resonator.kappa. Relax "
            "delay reads 5.0 * $qubit.t1. Open and Reset evaluate starting expressions "
            "once into direct input using captured sources. Inline qubit pulses read "
            "$qubit.freq and $qubit.wiring.L01.ch. Inline readout reads $resonator.freq, "
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
            "Proposes the SNR-maximizing readout frequency into parameter context "
            "$qubit.ext.best_ro_freq (MHz). When a cfg snapshot with pulse readout is "
            "available and $qubit.ext.best_ro_freq / $qubit.ext.best_ro_gain / "
            "$qubit.ext.best_ro_length are known from this result plus parameter context, "
            "also proposes $qubit.readout.readout_dpm."
        ),
        recommended=(
            "Analysis denoises the SNR curve before picking the peak. "
            "Wavelet smoothing is the default; switch to Gaussian only when "
            "you need to compare against the older sigma-based result. A "
            "'smooth' strength around 2 tames noise without washing out the "
            "feature."
        ),
    )

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> RoOptFreqRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        return RunRecord(cfg, FreqExp().run(cfg, context=context))

    def analyze(
        self,
        req: AnalyzeRequest[RoOptFreqRunResult, RoOptFreqAnalyzeParams],
        *,
        plots: Plots,
    ) -> RoOptFreqAnalyzeResult:
        params = req.analyze_params
        options = FreqAnalyzeOptions(
            smooth=params.smooth,
            smooth_method=params.smooth_method,
        )
        result = FreqExp().analyze(req.run_result, options, plots=plots)
        return RoOptFreqAnalyzeResult(
            best_freq=result.best_freq,
        )

    def get_writeback_items(
        self, req: WritebackRequest[RoOptFreqRunResult, RoOptFreqAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        items: list[WritebackItem] = [
            ParameterWriteback(
                target_name="$qubit.ext.best_ro_freq",
                description="Optimal readout frequency (MHz)",
                proposed_value=result.best_freq,
            ),
        ]
        items.extend(
            readout_dpm_writeback_items(
                req.ctx,
                req.run_result.cfg,
                proposed={"best_ro_freq": result.best_freq},
            )
        )
        return items

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_ro_opt_freq_{time.strftime('%m%d')}"
