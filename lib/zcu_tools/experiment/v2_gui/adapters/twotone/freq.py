from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Literal, TypeAlias

from matplotlib.figure import Figure

from zcu_tools.experiment.v2.twotone.freq import FreqCfg, FreqExp, FreqResult
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    CfgBuilder,
    build_exp_spec,
    make_pulse_module_spec,
    make_readout_module_spec,
    make_reset_module_spec,
    proper_qub_freq_range,
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

FreqRunResult: TypeAlias = FreqResult


@dataclass
class FreqAnalyzeParams:
    model_type: Annotated[Literal["lor", "sinc"], ParamMeta(label="Model type")]
    plot_fit: Annotated[bool, ParamMeta(label="Plot fit")]


@dataclass
class FreqAnalyzeResult(AnalyzeResultBase):
    freq: float
    freq_err: float
    fwhm: float
    fwhm_err: float
    params: dict[str, Any]
    figure: Figure


class FreqAdapter(
    BaseAdapter[FreqCfg, FreqRunResult, FreqAnalyzeResult, FreqAnalyzeParams]
):
    exp_cls = FreqExp
    ExpCfg_cls: ClassVar[Any] = FreqCfg
    legacy_migration_experiment: ClassVar[str | None] = "twotone/freq"

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Two-tone qubit spectroscopy: drives a qubit probe tone while "
            "reading out the resonator, sweeps the drive frequency, and fits "
            "the qubit response (Lorentzian or sinc) to extract the qubit "
            "transition frequency and its linewidth. Runs on real hardware. "
            "In first bring-up, run it after onetone/flux_dep has found "
            "'flx_int' and onetone/freq has re-calibrated the resonator at "
            "that flux."
        ),
        expects_md=(
            "Reads from the MetaDict (all optional): 'q_f' — qubit frequency, "
            "the sweep centre (~2000–6000 MHz); 'qf_w' — qubit linewidth, "
            "setting the half-span as 1.5*qf_w (~1–50 MHz); 'qub_ch' — "
            "qubit-drive channel; 'r_f' — resonator frequency for the readout "
            "tone (~4000–8000 MHz); 'res_ch' / 'ro_ch' — readout drive / ADC "
            "channels; 'timeFly' — readout trigger-offset cable delay (~0–1 "
            "us). Absent 'q_f'/'qf_w' → a fixed ±30 MHz span around 4000 MHz; "
            "for a first qubit search this default is only a placeholder, so "
            "override it with a broad, hardware-safe survey window."
        ),
        expects_ml=(
            "Needs a qubit-probe pulse module and a pulse-readout module; "
            "references a ModuleLibrary waveform named 'ro_waveform' for the "
            "readout shape when present. Optionally references a calibrated "
            "reset module (disabled when none exists)."
        ),
        typical_writeback=(
            "Proposes the fitted qubit frequency into MetaDict 'q_f' and the "
            "fitted linewidth (FWHM) into 'qf_w'. No ModuleLibrary writeback. "
            "Apply only after the agent/user verifies that the plotted feature "
            "is the intended qubit line."
        ),
        recommended=(
            "Standard flow at a fresh 'flx_int': first run a wide survey over "
            "the user-approved passband to find a real qubit feature, without "
            "using predictor output or simulator truth as the answer. Then "
            "run a narrower scan around the observed feature and fit it. "
            "Analysis defaults to the Lorentzian fit ('lor') with plot-fit on; "
            "switch to 'sinc' for power-broadened or saturated lines. If the "
            "scan looks like noise, widen/re-centre and check readout/gain "
            "before concluding there is no qubit."
        ),
    )

    @classmethod
    def cfg_spec(cls) -> CfgSectionSpec:
        return build_exp_spec(
            modules={
                "reset": make_reset_module_spec(optional=True),
                # The sweep axis owns the qubit-drive frequency
                # (set_param("freq") at run); lock it so the form does not
                # show a field the sweep silently overwrites.
                "qub_pulse": make_pulse_module_spec().lock_literal("freq", 0.0),
                "readout": make_readout_module_spec(),
            },
            sweep={"freq": SweepSpec(label="Freq (MHz)")},
        )

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        return (
            CfgBuilder(ctx, self.cfg_spec())
            .scalars(reps=1000, rounds=100, relax_delay=1.0)
            # optional → None (disabled) when no library reset (ADR-0010)
            .role("modules.reset", "reset", optional=True)
            .role("modules.qub_pulse", "qub_probe", prefer_blank=True)
            .role("modules.readout", "readout")
            .set_sweep("sweep.freq", proper_qub_freq_range(ctx, 301))
            .build()
        )

    def get_analyze_params(
        self, result: FreqRunResult, ctx: ExpContext
    ) -> FreqAnalyzeParams:
        return FreqAnalyzeParams(model_type="lor", plot_fit=True)

    def analyze(
        self, req: AnalyzeRequest[FreqRunResult, FreqAnalyzeParams]
    ) -> FreqAnalyzeResult:
        params = req.analyze_params
        freq, freq_err, fwhm, fwhm_err, fig = FreqExp().analyze(
            req.run_result,
            model_type=params.model_type,
            plot_fit=params.plot_fit,
        )
        return FreqAnalyzeResult(
            freq=freq,
            freq_err=freq_err,
            fwhm=fwhm,
            fwhm_err=fwhm_err,
            params={},
            figure=fig,
        )

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
