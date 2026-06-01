from __future__ import annotations

import time
from dataclasses import dataclass

from matplotlib.figure import Figure
from typing_extensions import Annotated, Any, ClassVar, Sequence, TypeAlias

from zcu_tools.experiment.v2.twotone.ro_optimize.freq_gain import (
    FreqGainCfg,
    FreqGainExp,
    FreqGainResult,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    make_pulse_module_spec,
    make_pulse_readout_module_spec,
    make_qub_probe_default,
    make_readout_default,
    make_reset_module_spec,
    make_reset_ref_default,
    proper_res_freq_range,
)
from zcu_tools.gui.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ExpContext,
    MetaDictWriteback,
    ParamMeta,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    WritebackItem,
    WritebackRequest,
)

RoOptFreqGainRunResult: TypeAlias = FreqGainResult


@dataclass
class RoOptFreqGainAnalyzeParams:
    smooth: Annotated[float, ParamMeta(label="Smooth sigma", decimals=2)]


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

    @classmethod
    def guide(cls) -> AdapterGuide:
        return AdapterGuide(
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
                "MetaDict 'best_ro_freq' (MHz) and 'best_ro_gain' (a.u.). No "
                "ModuleLibrary writeback — combine the best readout params into a "
                "'readout_dpm' module afterwards (the 'readout_dpm' role)."
            ),
            recommended=(
                "Analysis smooths the 2D SNR map before picking the peak; a "
                "'smooth' sigma around 1 is a reasonable default — raise it if the "
                "optimum jitters on a noisy map. Keep the freq span tight (a "
                "fraction of a linewidth around the known readout frequency) and "
                "the gain range modest so the 2D scan stays affordable."
            ),
        )

    @classmethod
    def cfg_spec(cls) -> CfgSectionSpec:
        return CfgSectionSpec(
            fields={
                "modules": CfgSectionSpec(
                    label="Modules",
                    fields={
                        "reset": make_reset_module_spec(optional=True),
                        "qub_pulse": make_pulse_module_spec(),
                        "readout": make_pulse_readout_module_spec(),
                    },
                ),
                "reps": ScalarSpec(label="Reps", type=int),
                "rounds": ScalarSpec(label="Rounds", type=int),
                "relax_delay": ScalarSpec(
                    label="Relax delay (us)", type=float, decimals=3
                ),
                "sweep": CfgSectionSpec(
                    label="Sweep",
                    fields={
                        "freq": SweepSpec(label="Readout freq (MHz)"),
                        "gain": SweepSpec(label="Readout gain (a.u.)"),
                    },
                ),
            }
        )

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        return CfgSectionValue(
            fields={
                "modules": CfgSectionValue(
                    fields={
                        "qub_pulse": make_qub_probe_default(ctx),
                        "readout": make_readout_default(ctx),
                        "reset": make_reset_ref_default(ctx, optional=True),
                    }
                ),
                "reps": DirectValue(100),
                "rounds": DirectValue(1000),
                "relax_delay": DirectValue(1.0),
                "sweep": CfgSectionValue(
                    fields={
                        "freq": proper_res_freq_range(ctx, 31, span_factor=0.5),
                        "gain": SweepValue(start=0.0, stop=0.2, expts=31),
                    },
                ),
            }
        )

    def get_analyze_params(
        self, result: RoOptFreqGainRunResult, ctx: ExpContext
    ) -> RoOptFreqGainAnalyzeParams:
        return RoOptFreqGainAnalyzeParams(smooth=1.0)

    def analyze(
        self, req: AnalyzeRequest[RoOptFreqGainRunResult, RoOptFreqGainAnalyzeParams]
    ) -> RoOptFreqGainAnalyzeResult:
        params = req.analyze_params
        best_freq, best_gain, fig = FreqGainExp().analyze(
            req.run_result, smooth=params.smooth
        )
        return RoOptFreqGainAnalyzeResult(
            best_freq=best_freq, best_gain=best_gain, figure=fig
        )

    def get_writeback_items(
        self, req: WritebackRequest[RoOptFreqGainRunResult, RoOptFreqGainAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        return [
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

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_ro_opt_freqgain_{time.strftime('%m%d')}"
