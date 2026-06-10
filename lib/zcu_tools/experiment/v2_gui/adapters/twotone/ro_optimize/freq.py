from __future__ import annotations

import time
from dataclasses import dataclass

from matplotlib.figure import Figure
from typing_extensions import Annotated, Any, ClassVar, Sequence, TypeAlias

from zcu_tools.experiment.v2.twotone.ro_optimize.freq import (
    FreqCfg,
    FreqExp,
    FreqResult,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    CfgBuilder,
    build_exp_spec,
    make_pulse_module_spec,
    make_pulse_readout_module_spec,
    make_reset_module_spec,
    proper_res_freq_range,
)
from zcu_tools.gui.app.main.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    CfgSectionSpec,
    CfgSectionValue,
    ExpContext,
    MetaDictWriteback,
    ParamMeta,
    SweepSpec,
    WritebackItem,
    WritebackRequest,
)

RoOptFreqRunResult: TypeAlias = FreqResult


@dataclass
class RoOptFreqAnalyzeParams:
    smooth: Annotated[float, ParamMeta(label="Smooth sigma", decimals=2)]


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

    @classmethod
    def guide(cls) -> AdapterGuide:
        return AdapterGuide(
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
                "'best_ro_freq' (MHz). No ModuleLibrary writeback — combine "
                "'best_ro_freq' / 'best_ro_gain' / 'best_ro_length' into a "
                "'readout_dpm' module afterwards (the 'readout_dpm' role)."
            ),
            recommended=(
                "Analysis smooths the SNR curve before picking the peak; a "
                "'smooth' sigma around 2 tames noise without washing out the "
                "feature — raise it if the optimum jitters, lower it for a sharp "
                "peak. A sweep of a couple of linewidths around 'r_f' usually "
                "brackets the best frequency."
            ),
        )

    @classmethod
    def cfg_spec(cls) -> CfgSectionSpec:
        return build_exp_spec(
            modules={
                "reset": make_reset_module_spec(optional=True),
                "qub_pulse": make_pulse_module_spec(),
                # The freq sweep owns the readout frequency
                # (set_param("freq") writes both pulse and ro freq), so
                # lock it off the form.
                "readout": make_pulse_readout_module_spec()
                .lock_literal("pulse_cfg.freq", 0.0)
                .lock_literal("ro_cfg.ro_freq", 0.0),
            },
            sweep={"freq": SweepSpec(label="Readout freq (MHz)")},
        )

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        return (
            CfgBuilder(ctx, self.cfg_spec())
            .scalars(reps=1000, rounds=100, relax_delay=1.0)
            .role("modules.qub_pulse", "qub_probe", prefer_blank=True)
            .role("modules.readout", "readout", prefer_blank=True)
            .role("modules.reset", "reset", optional=True)
            .set_sweep("sweep.freq", proper_res_freq_range(ctx, 301))
            .build()
        )

    def get_analyze_params(
        self, result: RoOptFreqRunResult, ctx: ExpContext
    ) -> RoOptFreqAnalyzeParams:
        return RoOptFreqAnalyzeParams(smooth=2.0)

    def analyze(
        self, req: AnalyzeRequest[RoOptFreqRunResult, RoOptFreqAnalyzeParams]
    ) -> RoOptFreqAnalyzeResult:
        params = req.analyze_params
        best_freq, fig = FreqExp().analyze(req.run_result, smooth=params.smooth)
        return RoOptFreqAnalyzeResult(best_freq=best_freq, figure=fig)

    def get_writeback_items(
        self, req: WritebackRequest[RoOptFreqRunResult, RoOptFreqAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        return [
            MetaDictWriteback(
                target_name="best_ro_freq",
                description="Optimal readout frequency (MHz)",
                proposed_value=result.best_freq,
            ),
        ]

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_ro_opt_freq_{time.strftime('%m%d')}"
