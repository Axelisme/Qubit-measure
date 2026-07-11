from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Literal, TypeAlias

from matplotlib.figure import Figure

from zcu_tools.experiment.v2.twotone.ro_optimize.freq import (
    FreqCfg,
    FreqExp,
    FreqResult,
)
from zcu_tools.experiment.v2_gui.adapters._support import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    readout_dpm_writeback_items,
    res_freq_range,
    scaled_md,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
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

RoOptFreqRunResult: TypeAlias = FreqResult


@dataclass
class RoOptFreqAnalyzeParams:
    smooth_method: Annotated[
        Literal["wavelet", "gaussian"], ParamMeta(label="Smooth method")
    ] = "wavelet"
    smooth: Annotated[float, ParamMeta(label="Smooth strength", decimals=2)] = 2.0


@dataclass
class RoOptFreqAnalyzeResult(AnalyzeResultBase):
    best_freq: float
    figure: Figure


class RoOptFreqAdapter(
    BaseAdapter[
        FreqCfg,
        RoOptFreqRunResult,
        RoOptFreqAnalyzeResult,
        RoOptFreqAnalyzeParams,
    ]
):
    exp_cls = FreqExp
    ExpCfg_cls: ClassVar[Any] = FreqCfg

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
            "Reads from the MetaDict (all optional): 'r_f' — resonator "
            "frequency, the sweep centre (~4000–8000 MHz); 'rf_w' — linewidth, "
            "setting the span as r_f ± 1.5*rf_w (~5–50 MHz; falls back to ±30 "
            "MHz when absent); 'res_ch' / 'ro_ch' — drive / ADC channels; "
            "'timeFly' — cable time-of-flight for the trigger offset; 'q_f' / "
            "'qub_ch' — qubit frequency / drive channel for the g↔e pi pulse."
        ),
        expects_ml=(
            "Needs a qubit-probe pulse module (typically a calibrated pi "
            "pulse, e.g. 'pi_amp') and a pulse-readout module (e.g. "
            "'readout_rf'); references a ModuleLibrary waveform named "
            "'ro_waveform' when present. Optionally references a reset module."
        ),
        typical_writeback=(
            "Proposes the SNR-maximizing readout frequency into MetaDict "
            "'best_ro_freq' (MHz). When a cfg snapshot with pulse readout is "
            "available and 'best_ro_freq' / 'best_ro_gain' / "
            "'best_ro_length' are known from this result plus MetaDict, also "
            "proposes ModuleLibrary 'readout_dpm'."
        ),
        recommended=(
            "Analysis denoises the SNR curve before picking the peak. "
            "Wavelet smoothing is the default; switch to Gaussian only when "
            "you need to compare against the older sigma-based result. A "
            "'smooth' strength around 2 tames noise without washing out the "
            "feature."
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
                locked={"pulse_cfg.freq": 0.0, "ro_cfg.ro_freq": 0.0},
            )
            .relax_delay(scaled_md("t1", factor=5.0, fallback_value=30.5))
            .sweep(
                "freq",
                label="Readout freq (MHz)",
                default=res_freq_range(expts=301, span_factor=1.0),
            )
            .float("skew_penalty", label="Skew penalty", default=0.0, decimals=3)
            .reps(1000)
            .rounds(100)
            .build()
        )

    def analyze(
        self, req: AnalyzeRequest[RoOptFreqRunResult, RoOptFreqAnalyzeParams]
    ) -> RoOptFreqAnalyzeResult:
        params = req.analyze_params
        best_freq, fig = FreqExp().analyze(
            req.run_result, smooth=params.smooth, smooth_method=params.smooth_method
        )
        return RoOptFreqAnalyzeResult(best_freq=best_freq, figure=fig)

    def get_writeback_items(
        self, req: WritebackRequest[RoOptFreqRunResult, RoOptFreqAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        items: list[WritebackItem] = [
            MetaDictWriteback(
                target_name="best_ro_freq",
                description="Optimal readout frequency (MHz)",
                proposed_value=result.best_freq,
            ),
        ]
        items.extend(
            readout_dpm_writeback_items(
                req.ctx,
                req.run_result.cfg_snapshot,
                proposed={"best_ro_freq": result.best_freq},
            )
        )
        return items

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_ro_opt_freq_{time.strftime('%m%d')}"
