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
from zcu_lab.v2.twotone.ro_optimize.freq_gain.core import (
    FreqGainAnalyzeOptions,
    FreqGainCfg,
    FreqGainExp,
    FreqGainResult,
)
from zcu_lab.v2.twotone.ro_optimize.freq_gain.form import FORM

RoOptFreqGainRunResult: TypeAlias = RunRecord[FreqGainCfg, FreqGainResult]


@dataclass
class RoOptFreqGainAnalyzeParams:
    smooth_method: Annotated[
        Literal["wavelet", "gaussian"], ParamMeta(label="Smooth method")
    ] = "wavelet"
    smooth: Annotated[float, ParamMeta(label="Smooth strength", decimals=2)] = 1.0


@dataclass
class RoOptFreqGainAnalyzeResult(AnalyzeResultBase):
    best_freq: float
    best_gain: float


class RoOptFreqGainAdapter(
    BaseAdapter[
        FreqGainCfg,
        RoOptFreqGainRunResult,
        RoOptFreqGainAnalyzeResult,
        RoOptFreqGainAnalyzeParams,
    ]
):
    experiment_tag: ClassVar[str | None] = FreqGainExp.persistence_tag()
    exp_cls = FreqGainExp
    ExpCfg_cls = FreqGainCfg
    form = FORM

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Readout frequency–power joint optimization: with the qubit "
            "toggled between g and e by a pi pulse, runs a 2D sweep of readout "
            "frequency × readout gain and measures the g/e signal-to-noise "
            "ratio (SNR), picking the (freq, gain) pair that best resolves the "
            "states. Runs on real hardware. Use this to refine freq and power "
            "together once you have a rough readout frequency."
        ),
        expects_md=(
            "Seeds 31 frequency points at $resonator.freq +/- 0.5 * $resonator.kappa, not "
            "$qubit.ext.best_ro_freq. Seeds 31 gain points from 0.0 to 0.2. Relax delay "
            "reads 5.0 * $qubit.t1. Open and Reset evaluate starting expressions once "
            "into direct input using captured sources. Inline qubit pulses read "
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
            "Proposes the SNR-maximizing readout frequency and gain into parameter "
            "context $qubit.ext.best_ro_freq (MHz) and $qubit.ext.best_ro_gain (a.u.). "
            "When a cfg snapshot with pulse readout is available and "
            "$qubit.ext.best_ro_freq / $qubit.ext.best_ro_gain / "
            "$qubit.ext.best_ro_length are known from this result plus parameter context, "
            "also proposes $qubit.readout.readout_dpm."
        ),
        recommended=(
            "Analysis denoises the 2D SNR map before picking the peak. Wavelet "
            "smoothing is the default; switch to Gaussian only when comparing "
            "against the older sigma-based result. Keep the freq span tight and "
            "the gain range modest so the 2D scan stays affordable."
        ),
    )

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> RoOptFreqGainRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        return RunRecord(cfg, FreqGainExp().run(cfg, context=context))

    def analyze(
        self,
        req: AnalyzeRequest[RoOptFreqGainRunResult, RoOptFreqGainAnalyzeParams],
        *,
        plots: Plots,
    ) -> RoOptFreqGainAnalyzeResult:
        params = req.analyze_params
        options = FreqGainAnalyzeOptions(
            smooth=params.smooth,
            smooth_method=params.smooth_method,
        )
        result = FreqGainExp().analyze(req.run_result, options, plots=plots)
        return RoOptFreqGainAnalyzeResult(
            best_freq=result.best_freq,
            best_gain=result.best_gain,
        )

    def get_writeback_items(
        self, req: WritebackRequest[RoOptFreqGainRunResult, RoOptFreqGainAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        items: list[WritebackItem] = [
            ParameterWriteback(
                target_name="best_ro_freq",
                description="Optimal readout frequency (MHz)",
                proposed_value=result.best_freq,
            ),
            ParameterWriteback(
                target_name="best_ro_gain",
                description="Optimal readout gain (a.u.)",
                proposed_value=result.best_gain,
            ),
        ]
        items.extend(
            readout_dpm_writeback_items(
                req.ctx,
                req.run_result.cfg,
                proposed={
                    "best_ro_freq": result.best_freq,
                    "best_ro_gain": result.best_gain,
                },
            )
        )
        return items

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_ro_opt_freqgain_{time.strftime('%m%d')}"
