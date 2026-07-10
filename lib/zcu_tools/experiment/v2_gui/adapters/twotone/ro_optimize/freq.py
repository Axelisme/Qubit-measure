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
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    CfgBuilder,
    RoleInit,
    build_exp_spec,
    make_pulse_module_spec,
    make_pulse_readout_module_spec,
    make_reset_module_spec,
    proper_relax,
    proper_res_freq_range,
    readout_dpm_writeback_items,
)
from zcu_tools.gui.app.main.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    CfgSectionSpec,
    CfgSectionValue,
    ExpContext,
    FloatSpec,
    MetaDictWriteback,
    ParamMeta,
    SweepSpec,
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
    def cfg_spec(cls) -> CfgSectionSpec:
        return build_exp_spec(
            modules={
                "reset": make_reset_module_spec(optional=True),
                "qub_pulse": make_pulse_module_spec(),
                "readout": make_pulse_readout_module_spec()
                .lock_literal("pulse_cfg.freq", 0.0)
                .lock_literal("ro_cfg.ro_freq", 0.0),
            },
            sweep={"freq": SweepSpec(label="Readout freq (MHz)")},
            extra={"skew_penalty": FloatSpec(label="Skew penalty", decimals=3)},
        )

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        return (
            CfgBuilder(ctx, self.cfg_spec())
            .scalars(
                reps=1000,
                rounds=100,
                relax_delay=proper_relax(ctx, fallback=30.5),
                skew_penalty=0.0,
            )
            .role("modules.reset", "reset", RoleInit.DISABLED)
            .role("modules.qub_pulse", "pi_pulse")
            .role("modules.readout", "readout")
            .sweep("sweep.freq", proper_res_freq_range(ctx, 301, span_factor=1.0))
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
