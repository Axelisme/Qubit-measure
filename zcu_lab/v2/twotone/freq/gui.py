from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Literal, TypeAlias

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
    SampleWriteback,
    SessionEnv,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.twotone.freq.core import (
    FreqAnalyzeOptions,
    FreqCfg,
    FreqExp,
    FreqResult,
)
from zcu_lab.v2.twotone.freq.form import FORM

FreqRunResult: TypeAlias = RunRecord[FreqCfg, FreqResult]


@dataclass
class FreqAnalyzeParams:
    model_type: Annotated[Literal["lor", "sinc"], ParamMeta(label="Model type")] = "lor"
    plot_fit: Annotated[bool, ParamMeta(label="Plot fit")] = True


@dataclass
class FreqAnalyzeResult(AnalyzeResultBase):
    freq: float
    freq_err: float
    fwhm: float
    fwhm_err: float


class FreqAdapter(
    BaseAdapter[FreqCfg, FreqRunResult, FreqAnalyzeResult, FreqAnalyzeParams]
):
    experiment_tag: ClassVar[str | None] = FreqExp.persistence_tag()
    exp_cls = FreqExp
    ExpCfg_cls: ClassVar[Any] = FreqCfg
    form = FORM

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Two-tone qubit spectroscopy: drives a qubit probe tone while "
            "reading out the resonator, sweeps the drive frequency, and fits "
            "the qubit response (Lorentzian or sinc) to extract the qubit "
            "transition frequency and its linewidth. Runs on real hardware. "
            "In first bring-up, run it after onetone/flux_dep has found "
            "$qubit.flux_int and onetone/freq has re-calibrated the resonator at "
            "that flux."
        ),
        expects_md=(
            "Opening a tab or Reset evaluates Form starting expressions once "
            "against captured parameters into direct input. Range.around seeds "
            "301 points at $qubit.freq ± 1.5 * $qubit.kappa, in MHz. The fresh "
            "probe reads $qubit.freq and $qubit.wiring.L01.ch; its gain is 0.1 "
            "and waveform length is 5.0 us. The inline readout fallback reads "
            "$resonator.freq, $resonator.wiring.ch and $resonator.wiring.ro_ch "
            "for its tones and channels, and $resonator.ext.time_of_fly + "
            "0.05 us for trigger timing. Missing or invalid required expression "
            "sources leave the affected fields empty and Invalid. Supply valid "
            "calibration or direct values before Run. There is no numeric "
            "frequency-survey fallback; for a first qubit search, review the "
            "hardware-safe passband and enter a user-approved wide sweep."
        ),
        expects_ml=(
            "Starts with a fresh inline 'qub_probe' const pulse, not a calibrated "
            "state-prep pulse. $qubit.module.readout resolves to a captured "
            "library link; an unset/dangling slot or unbound alias uses a fresh "
            "inline 'readout' template with a const waveform. That fallback "
            "still needs its required parameter sources. $qubit.module.reset "
            "is optional; an unset/dangling slot or unbound alias leaves Reset "
            "disabled and Valid. Init Pulse stays disabled."
        ),
        typical_writeback=(
            "Proposes fitted qubit.freq and qubit.kappa (FWHM) in MHz. An "
            "independent freq_mhz/freq_err_mhz sample uses the captured Run "
            "bias when device evidence is available. No module writeback. "
            "Apply only after the agent/user verifies that the plotted feature "
            "is the intended qubit line."
        ),
        recommended=(
            "Standard flow at a fresh $qubit.flux_int: first run a wide survey over "
            "the user-approved passband to find a real qubit feature, without "
            "using predictor output or simulator truth as the answer. Then "
            "run a narrower scan around the observed feature and fit it. "
            "Analysis defaults to the Lorentzian fit ('lor') with plot-fit on; "
            "switch to 'sinc' for power-broadened or saturated lines. If the "
            "scan looks like noise, widen/re-centre and check readout/gain "
            "before concluding there is no qubit."
        ),
    )

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> FreqRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        result = FreqExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self, req: AnalyzeRequest[FreqRunResult, FreqAnalyzeParams], *, plots: Plots
    ) -> FreqAnalyzeResult:
        params = req.analyze_params
        analysis = FreqExp().analyze(
            req.run_result,
            FreqAnalyzeOptions(model_type=params.model_type, plot_fit=params.plot_fit),
            plots=plots,
        )
        return FreqAnalyzeResult(
            freq=analysis.freq,
            freq_err=analysis.freq_err,
            fwhm=analysis.fwhm,
            fwhm_err=analysis.fwhm_err,
        )

    def get_writeback_items(
        self, req: WritebackRequest[FreqRunResult, FreqAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        """Propose fitted qubit frequency parameters and a captured-component sample.

        req supplies the source result and fixed Run context. Sample fields use
        MHz and target the Run's resolved $qubit, never the live selection. Without
        a parameter capture, retain parameter proposals only so loaded data can
        still be analyzed; Apply rejects the missing capture. Unbound $qubit
        with a capture raises RoleResolutionError. WritebackService supplies
        bias coordinates or omits the sample with its existing public reason.
        No write or device access occurs here; this Run leaves devices fixed.
        """
        result = req.analyze_result
        items: list[WritebackItem] = [
            ParameterWriteback(
                target_name="$qubit.freq",
                description="Qubit frequency (MHz)",
                proposed_value=result.freq,
            ),
            ParameterWriteback(
                target_name="$qubit.kappa",
                description="Qubit linewidth FWHM (MHz)",
                proposed_value=result.fwhm,
            ),
        ]
        parameters = req.ctx.parameters
        if parameters is not None:
            items.append(
                SampleWriteback(
                    target_name=parameters.resolve_path("$qubit"),
                    description="Qubit frequency sample at the captured bias",
                    proposed_value={
                        "freq_mhz": result.freq,
                        "freq_err_mhz": result.freq_err,
                    },
                )
            )
        return items

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_qubit_freq_{time.strftime('%m%d')}"
