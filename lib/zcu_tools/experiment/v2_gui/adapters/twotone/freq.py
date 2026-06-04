from __future__ import annotations

import time
from dataclasses import dataclass

from matplotlib.figure import Figure
from typing_extensions import Annotated, Any, ClassVar, Literal, Sequence, TypeAlias

from zcu_tools.experiment.v2.twotone.freq import FreqCfg, FreqExp, FreqResult
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    make_pulse_module_spec,
    make_qub_probe_default,
    make_readout_default,
    make_readout_module_spec,
    make_reset_module_spec,
    make_reset_ref_default,
    proper_qub_freq_range,
)
from zcu_tools.gui.app.main.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    CfgNodeValue,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ExpContext,
    MetaDictWriteback,
    ParamMeta,
    ScalarSpec,
    SweepSpec,
    WritebackItem,
    WritebackRequest,
)

FreqRunResult: TypeAlias = FreqResult


@dataclass
class FreqAnalyzeParams:
    model_type: Annotated[Literal["lor", "sinc"], ParamMeta(label="Model type")]
    plot_fit: Annotated[bool, ParamMeta(label="Plot fit")]


@dataclass
class FreqAnalyzeResult(AnalyzeResultBase):
    freq: float
    fwhm: float
    params: dict[str, Any]
    figure: Figure


class FreqAdapter(
    BaseAdapter[FreqCfg, FreqRunResult, FreqAnalyzeResult, FreqAnalyzeParams]
):
    exp_cls = FreqExp
    ExpCfg_cls: ClassVar[Any] = FreqCfg

    @classmethod
    def guide(cls) -> AdapterGuide:
        return AdapterGuide(
            behavior=(
                "Two-tone qubit spectroscopy: drives a qubit probe tone while "
                "reading out the resonator, sweeps the drive frequency, and fits "
                "the qubit response (Lorentzian or sinc) to extract the qubit "
                "transition frequency and its linewidth. Runs on real hardware. "
                "Typically run once the resonator is calibrated and you have a "
                "rough idea of where the qubit sits."
            ),
            expects_md=(
                "Reads from the MetaDict (all optional): 'q_f' — qubit frequency, "
                "the sweep centre (~2000–6000 MHz); 'qf_w' — qubit linewidth, "
                "setting the half-span as 1.5*qf_w (~1–50 MHz); 'qub_ch' — "
                "qubit-drive channel; 'r_f' — resonator frequency for the readout "
                "tone (~4000–8000 MHz); 'res_ch' / 'ro_ch' — readout drive / ADC "
                "channels; 'timeFly' — readout trigger-offset cable delay (~0–1 "
                "us). Absent 'q_f'/'qf_w' → a fixed ±30 MHz span around 4000 MHz."
            ),
            expects_ml=(
                "Needs a qubit-probe pulse module and a pulse-readout module; "
                "references a ModuleLibrary waveform named 'ro_waveform' for the "
                "readout shape when present. Optionally references a calibrated "
                "reset module (disabled when none exists)."
            ),
            typical_writeback=(
                "Proposes the fitted qubit frequency into MetaDict 'q_f' and the "
                "fitted linewidth (FWHM) into 'qf_w'. No ModuleLibrary writeback."
            ),
            recommended=(
                "Analysis defaults to the Lorentzian fit ('lor') with plot-fit on; "
                "switch to 'sinc' for power-broadened or saturated lines. A sweep "
                "of ~301 points spanning a couple of linewidths around the "
                "expected qubit frequency captures the peak cleanly; widen the "
                "span and lower the drive gain if the qubit has drifted or the "
                "line is washed out."
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
                        "readout": make_readout_module_spec(),
                    },
                ),
                "reps": ScalarSpec(label="Reps", type=int),
                "rounds": ScalarSpec(label="Rounds", type=int),
                "relax_delay": ScalarSpec(
                    label="Relax delay (us)", type=float, decimals=3
                ),
                "sweep": CfgSectionSpec(
                    label="Sweep",
                    fields={"freq": SweepSpec(label="Freq (MHz)")},
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
                        # optional → DisabledRefValue when no library reset (ADR-0012)
                        "reset": make_reset_ref_default(ctx, optional=True),
                    }
                ),
                "reps": DirectValue(100),
                "rounds": DirectValue(100),
                "relax_delay": DirectValue(1.0),
                "sweep": CfgSectionValue(
                    fields={"freq": proper_qub_freq_range(ctx, 301)},
                ),
            }
        )

    def get_analyze_params(
        self, result: FreqRunResult, ctx: ExpContext
    ) -> FreqAnalyzeParams:
        return FreqAnalyzeParams(model_type="lor", plot_fit=True)

    def analyze(
        self, req: AnalyzeRequest[FreqRunResult, FreqAnalyzeParams]
    ) -> FreqAnalyzeResult:
        params = req.analyze_params
        freq, fwhm, fig = FreqExp().analyze(
            req.run_result,
            model_type=params.model_type,
            plot_fit=params.plot_fit,
        )
        return FreqAnalyzeResult(freq=freq, fwhm=fwhm, params={}, figure=fig)

    def get_writeback_items(
        self, req: WritebackRequest[FreqRunResult, FreqAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        return [
            MetaDictWriteback(
                target_name="q_f",
                description="Qubit frequency (MHz)",
                proposed_value=result.freq,
            ),
            MetaDictWriteback(
                target_name="qf_w",
                description="Qubit linewidth FWHM (MHz)",
                proposed_value=result.fwhm,
            ),
        ]

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_qubit_freq_{time.strftime('%m%d')}"
