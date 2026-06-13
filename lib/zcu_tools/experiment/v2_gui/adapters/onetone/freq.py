from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Literal, TypeAlias

from matplotlib.figure import Figure

from zcu_tools.experiment.v2.onetone.freq import FreqCfg, FreqExp, FreqResult
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    CfgBuilder,
    build_exp_spec,
    make_pulse_readout_module_spec,
    md_get_float,
    md_has_key,
    proper_res_freq_range,
)
from zcu_tools.gui.app.main.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    CfgSectionSpec,
    CfgSectionValue,
    EvalValue,
    ExpContext,
    MetaDictWriteback,
    ParamMeta,
    SweepSpec,
    WritebackItem,
    WritebackRequest,
)

OneToneFreqRunResult: TypeAlias = FreqResult


@dataclass
class OneToneFreqAnalyzeParams:
    model_type: Annotated[Literal["hm", "t", "auto"], ParamMeta(label="Model type")]
    fit_bg_slope: Annotated[bool, ParamMeta(label="Fit background slope")]


@dataclass
class OneToneFreqAnalyzeResult(AnalyzeResultBase):
    freq: float
    fwhm: float
    params: dict[str, Any]
    figure: Figure


class OneToneFreqAdapter(
    BaseAdapter[
        FreqCfg,
        OneToneFreqRunResult,
        OneToneFreqAnalyzeResult,
        OneToneFreqAnalyzeParams,
    ]
):
    exp_cls = FreqExp
    ExpCfg_cls: ClassVar[Any] = FreqCfg

    @classmethod
    def guide(cls) -> AdapterGuide:
        return AdapterGuide(
            behavior=(
                "One-tone resonator spectroscopy: sweeps the readout frequency "
                "and fits the resonator response to extract its frequency and "
                "linewidth. Runs on real hardware. Typically run after a coarse "
                "resonator search has placed you near the right frequency."
            ),
            expects_md=(
                "Reads from the MetaDict (all optional): 'r_f' — resonator "
                "frequency (~4000–8000 MHz); 'res_probe_len' — probe-pulse length, "
                "from which the readout window is derived (~0.5–5 us); 'res_ch' / "
                "'ro_ch' — drive / readout channel indices; 'timeFly' — cable "
                "time-of-flight feeding the trigger offset (~0–1 us)."
            ),
            expects_ml=(
                "Needs a pulse-readout module, and references a ModuleLibrary "
                "waveform named 'ro_waveform' when one exists (optional)."
            ),
            typical_writeback=(
                "Proposes the fitted resonator frequency and linewidth into "
                "MetaDict 'r_f' / 'rf_w'. The readout module / waveform are left "
                "to the user — a frequency fit alone does not justify rewriting "
                "the whole readout config."
            ),
            recommended=(
                "Analysis defaults to the hanger-model fit ('hm') with "
                "background-slope fitting on. A sweep spanning a few linewidths "
                "around the known resonator frequency usually captures the dip "
                "cleanly; widen it if the resonator has drifted."
            ),
        )

    @classmethod
    def cfg_spec(cls) -> CfgSectionSpec:
        return build_exp_spec(
            modules={
                # No reset module — one-tone spectroscopy runs without a
                # qubit reset (the ExpCfg defaults reset=None).
                "readout": make_pulse_readout_module_spec()
                .lock_literal("pulse_cfg.freq", 0.0)
                .lock_literal("ro_cfg.ro_freq", 0.0),
            },
            sweep={"freq": SweepSpec(label="Freq (MHz)")},
        )

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        probe_len = md_get_float(ctx, "res_probe_len", 1.0)
        ro_length: float | EvalValue = (
            EvalValue(expr="res_probe_len - 0.1")
            if md_has_key(ctx, "res_probe_len")
            else probe_len - 0.1
        )
        return (
            CfgBuilder(ctx, self.cfg_spec())
            .scalars(reps=100, rounds=100, relax_delay=1.0)
            # cfg_spec() locks pulse_cfg.freq / ro_cfg.ro_freq to 0.0 (the sweep
            # axis owns frequency); build() fills those locked literals from the
            # spec, so they are not set here.
            .role("modules.readout", "pulse_readout")
            .set("modules.readout.pulse_cfg.gain", 0.05)
            .set("modules.readout.ro_cfg.ro_length", ro_length)
            .set_sweep("sweep.freq", proper_res_freq_range(ctx, 301))
            .build()
        )

    def get_analyze_params(
        self, result: OneToneFreqRunResult, ctx: ExpContext
    ) -> OneToneFreqAnalyzeParams:
        return OneToneFreqAnalyzeParams(model_type="hm", fit_bg_slope=True)

    def analyze(
        self, req: AnalyzeRequest[OneToneFreqRunResult, OneToneFreqAnalyzeParams]
    ) -> OneToneFreqAnalyzeResult:
        params = req.analyze_params
        freq, fwhm, fit_params, figure = FreqExp().analyze(
            req.run_result,
            model_type=params.model_type,
            fit_bg_slope=params.fit_bg_slope,
        )
        return OneToneFreqAnalyzeResult(
            freq=freq,
            fwhm=fwhm,
            params=fit_params,
            figure=figure,
        )

    def get_writeback_items(
        self, req: WritebackRequest[OneToneFreqRunResult, OneToneFreqAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        return [
            MetaDictWriteback(
                target_name="r_f",
                description="Resonator frequency (MHz)",
                proposed_value=result.freq,
            ),
            MetaDictWriteback(
                target_name="rf_w",
                description="Resonator linewidth FWHM (MHz)",
                proposed_value=result.fwhm,
            ),
        ]

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.res_name}_freq_{time.strftime('%m%d')}"
