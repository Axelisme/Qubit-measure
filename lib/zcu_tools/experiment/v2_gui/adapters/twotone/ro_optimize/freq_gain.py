from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Literal, TypeAlias

from matplotlib.figure import Figure

from zcu_tools.experiment.v2.twotone.ro_optimize.freq_gain import (
    FreqGainCfg,
    FreqGainExp,
    FreqGainResult,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    best_ro_freq_range,
    readout_dpm_writeback_items,
    scaled_md,
)
from zcu_tools.gui.app.main.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    ExpContext,
    MetaDictWriteback,
    ParamMeta,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.cfg import (
    SweepValue,
)

RoOptFreqGainRunResult: TypeAlias = FreqGainResult


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
    figure: Figure


class RoOptFreqGainAdapter(
    BaseAdapter[
        FreqGainCfg,
        RoOptFreqGainRunResult,
        RoOptFreqGainAnalyzeResult,
        RoOptFreqGainAnalyzeParams,
    ]
):
    exp_cls = FreqGainExp
    ExpCfg_cls: ClassVar[Any] = FreqGainCfg

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
            "Reads from the MetaDict (all optional): 'r_f' / 'best_ro_freq' — "
            "resonator / chosen readout frequency centring the freq sweep "
            "(~4000–8000 MHz); 'rf_w' — linewidth, setting the freq half-span "
            "(~5–50 MHz); 'res_ch' / 'ro_ch' — drive / ADC channels; 'timeFly' "
            "— trigger-offset cable delay; 'q_f' / 'qub_ch' — qubit frequency "
            "/ channel for the g↔e pi pulse."
        ),
        expects_ml=(
            "Needs a qubit-probe pulse module (typically a calibrated pi "
            "pulse, e.g. 'pi_amp') and a pulse-readout module (e.g. "
            "'readout_rf'); references a ModuleLibrary waveform 'ro_waveform' "
            "when present. Optionally references a reset module."
        ),
        typical_writeback=(
            "Proposes the SNR-maximizing readout frequency and gain into "
            "MetaDict 'best_ro_freq' (MHz) and 'best_ro_gain' (a.u.). When a "
            "cfg snapshot with pulse readout is available and "
            "'best_ro_freq' / 'best_ro_gain' / 'best_ro_length' are known "
            "from this result plus MetaDict, also proposes ModuleLibrary "
            "'readout_dpm'."
        ),
        recommended=(
            "Analysis denoises the 2D SNR map before picking the peak. Wavelet "
            "smoothing is the default; switch to Gaussian only when comparing "
            "against the older sigma-based result. Keep the freq span tight and "
            "the gain range modest so the 2D scan stays affordable."
        ),
    )

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder()
            .reset(optional=True)
            .pulse("qub_pulse", role_id="pi_pulse")
            .readout(
                pulse_only=True,
                locked={
                    "pulse_cfg.freq": 0.0,
                    "ro_cfg.ro_freq": 0.0,
                    "pulse_cfg.gain": 0.0,
                },
            )
            .relax_delay(scaled_md("t1", factor=5.0, fallback_value=100.0))
            .sweep(
                "freq",
                label="Readout freq (MHz)",
                default=best_ro_freq_range(expts=31, span_factor=0.5),
            )
            .sweep(
                "gain",
                label="Readout gain (a.u.)",
                default=SweepValue(start=0.0, stop=0.2, expts=31),
            )
            .float("skew_penalty", label="Skew penalty", default=0.0, decimals=3)
            .reps(100)
            .rounds(1000)
            .build()
        )

    def analyze(
        self, req: AnalyzeRequest[RoOptFreqGainRunResult, RoOptFreqGainAnalyzeParams]
    ) -> RoOptFreqGainAnalyzeResult:
        params = req.analyze_params
        best_freq, best_gain, fig = FreqGainExp().analyze(
            req.run_result, smooth=params.smooth, smooth_method=params.smooth_method
        )
        return RoOptFreqGainAnalyzeResult(
            best_freq=best_freq, best_gain=best_gain, figure=fig
        )

    def get_writeback_items(
        self, req: WritebackRequest[RoOptFreqGainRunResult, RoOptFreqGainAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        items: list[WritebackItem] = [
            MetaDictWriteback(
                target_name="best_ro_freq",
                description="Optimal readout frequency (MHz)",
                proposed_value=result.best_freq,
            ),
            MetaDictWriteback(
                target_name="best_ro_gain",
                description="Optimal readout gain (a.u.)",
                proposed_value=result.best_gain,
            ),
        ]
        items.extend(
            readout_dpm_writeback_items(
                req.ctx,
                req.run_result.cfg_snapshot,
                proposed={
                    "best_ro_freq": result.best_freq,
                    "best_ro_gain": result.best_gain,
                },
            )
        )
        return items

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_ro_opt_freqgain_{time.strftime('%m%d')}"
