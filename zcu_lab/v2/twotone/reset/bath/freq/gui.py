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

from zcu_lab.v2.twotone.reset.bath.freq.core import (
    FreqGainAnalyzeOptions,
    FreqGainCfg,
    FreqGainExp,
    FreqGainResult,
)
from zcu_lab.v2.twotone.reset.bath.freq.form import FORM

BathFreqGainRunResult: TypeAlias = RunRecord[FreqGainCfg, FreqGainResult]


@dataclass
class BathFreqGainAnalyzeParams:
    smooth_method: Annotated[
        Literal["wavelet", "gaussian"], ParamMeta(label="Smooth method")
    ] = "wavelet"
    smooth: Annotated[float, ParamMeta(label="Smooth strength", decimals=2)] = 1.0


@dataclass
class BathFreqGainAnalyzeResult(AnalyzeResultBase):
    gain: float
    freq: float


class BathFreqGainAdapter(
    BaseAdapter[
        FreqGainCfg,
        BathFreqGainRunResult,
        BathFreqGainAnalyzeResult,
        BathFreqGainAnalyzeParams,
    ]
):
    experiment_tag: ClassVar[str | None] = FreqGainExp.persistence_tag()
    exp_cls = FreqGainExp
    ExpCfg_cls = FreqGainCfg
    form = FORM

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Bath-reset cavity frequency–gain map: a 2D sweep of the bath "
            "reset's cavity-tone frequency × gain, with an internal 4-point "
            "pi/2 tomography phase axis the experiment drives itself, imaging "
            "the residual coherence to locate the (freq, gain) pair that best "
            "dumps the excitation into the cavity bath. Runs on real hardware."
        ),
        expects_md=(
            "Seeds 51 frequency points from $resonator.freq - 1.2 * $qubit.ext.rabi_freq "
            "to $resonator.freq - 0.8 * $qubit.ext.rabi_freq, and 51 gain points from 0.4 "
            "to 1.0. Relax delay reads 5.0 * $qubit.t1. The pi/2 phase offset stays "
            "locked at 90 degrees; Run adds its tomography phase cycle. Open and Reset "
            "evaluate starting expressions once into direct input using captured sources. "
            "Inline qubit pulses read $qubit.freq and $qubit.wiring.L01.ch. Inline "
            "readout reads $resonator.freq, $resonator.wiring.ch, $resonator.wiring.ro_ch "
            "and $resonator.ext.time_of_fly + 0.05 us for trigger timing. Missing or "
            "invalid required sources leave the affected fields empty and Invalid; supply "
            "calibration or direct values before Run. No numeric calibration fallback is "
            "supplied."
        ),
        expects_ml=(
            "Tested reset starts inline from the bath_reset template. Optional init pulse "
            "uses $qubit.module.x180 without fallback; an unset/dangling slot or unbound "
            "alias leaves it disabled and Valid. The pi/2 tone uses $qubit.module.x90, "
            "otherwise a fresh pi2_pulse template. It is copied inline so the fixed "
            "90-degree offset can be applied without changing the library. "
            "$qubit.module.readout selects a captured library link, otherwise a fresh "
            "inline readout template with a const waveform. Its required parameter "
            "sources must still be present. Optional $qubit.module.reset has no fallback: "
            "an unset/dangling slot or unbound alias leaves it disabled and Valid."
        ),
        typical_writeback=(
            "Proposes the best cavity gain and frequency to "
            "$qubit.ext.reset_bath_gain and $qubit.ext.reset_bath_freq only. "
            "BathPhase proposes the tested ground/excited reset modules; "
            "length analysis has no automatic writeback."
        ),
        recommended=(
            "Analysis denoises the 2D map before picking the peak; wavelet "
            "smoothing is the default and Gaussian remains available for "
            "comparison. Save writes one 3D HDF5 file with the tomography phase "
            "stored as an internal sweep axis."
        ),
    )

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> BathFreqGainRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        result = FreqGainExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self,
        req: AnalyzeRequest[BathFreqGainRunResult, BathFreqGainAnalyzeParams],
        *,
        plots: Plots,
    ) -> BathFreqGainAnalyzeResult:
        params = req.analyze_params
        analysis = FreqGainExp().analyze(
            req.run_result,
            FreqGainAnalyzeOptions(
                smooth=params.smooth, smooth_method=params.smooth_method
            ),
            plots=plots,
        )
        return BathFreqGainAnalyzeResult(gain=analysis.gain, freq=analysis.freq)

    def get_writeback_items(
        self,
        req: WritebackRequest[BathFreqGainRunResult, BathFreqGainAnalyzeResult],
    ) -> Sequence[WritebackItem]:
        """Propose this analysis's cavity gain then frequency to qubit ext."""
        result = req.analyze_result
        items: list[WritebackItem] = [
            ParameterWriteback(
                target_name="$qubit.ext.reset_bath_gain",
                description="Bath-reset cavity gain (a.u.)",
                proposed_value=result.gain,
            ),
            ParameterWriteback(
                target_name="$qubit.ext.reset_bath_freq",
                description="Bath-reset cavity frequency (MHz)",
                proposed_value=result.freq,
            ),
        ]
        return items

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_bathreset_freqgain_{time.strftime('%m%d')}"
