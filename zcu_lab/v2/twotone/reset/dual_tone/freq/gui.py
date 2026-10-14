from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal, TypeAlias

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
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.twotone.reset.dual_tone.freq.core import (
    FreqAnalyzeOptions,
    FreqCfg,
    FreqExp,
    FreqResult,
)
from zcu_lab.v2.twotone.reset.dual_tone.freq.form import make_form

DualToneFreqRunResult: TypeAlias = RunRecord[FreqCfg, FreqResult]


@dataclass
class DualToneFreqAnalyzeParams:
    smooth_method: Annotated[
        Literal["wavelet", "gaussian"], ParamMeta(label="Smooth method")
    ] = "wavelet"
    smooth: Annotated[float, ParamMeta(label="Smooth strength", decimals=2)] = 1.0


@dataclass
class DualToneFreqAnalyzeResult(AnalyzeResultBase):
    freq1: float
    freq2: float


class DualToneFreqAdapter(
    BaseAdapter[
        FreqCfg,
        DualToneFreqRunResult,
        DualToneFreqAnalyzeResult,
        DualToneFreqAnalyzeParams,
    ]
):
    experiment_tag: ClassVar[str | None] = FreqExp.persistence_tag()
    exp_cls = FreqExp

    @staticmethod
    def make_form(ctx: FormContext) -> Form[FreqCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Dual-tone reset frequency map: a 2D sweep of the two reset-tone "
            "frequencies (pulse1 × pulse2 of the tested two-pulse reset), "
            "imaging the qubit response to locate the pair of sideband "
            "frequencies that jointly dump the excitation. Runs on real "
            "hardware in hard-sweep mode (both axes are QICK register sweeps)."
        ),
        expects_md=(
            "Seeds two 201-point axes at $qubit.ext.reset_f1 and $qubit.ext.reset_f2, "
            "each +/- 50 MHz. Tested gains read $qubit.ext.reset_gain1 and "
            "$qubit.ext.reset_gain2. Relax delay is 0.5 us. Open and Reset evaluate "
            "starting expressions once into direct input using captured sources. Inline "
            "qubit pulses read $qubit.freq and $qubit.wiring.L01.ch. Inline readout reads "
            "$resonator.freq, $resonator.wiring.ch, $resonator.wiring.ro_ch and "
            "$resonator.ext.time_of_fly + 0.05 us for trigger timing. Missing or invalid "
            "required sources leave the affected fields empty and Invalid; supply "
            "calibration or direct values before Run. No numeric calibration fallback is "
            "supplied."
        ),
        expects_ml=(
            "Tested reset starts inline from the two_pulse_reset template. Optional init "
            "pulse uses $qubit.module.x180 without fallback; an unset/dangling slot or "
            "unbound alias leaves it disabled and Valid. $qubit.module.readout selects a "
            "captured library link, otherwise a fresh inline readout template with a "
            "const waveform. Its required parameter sources must still be present. "
            "Optional $qubit.module.reset has no fallback: an unset/dangling slot or "
            "unbound alias leaves it disabled and Valid."
        ),
        typical_writeback=(
            "Proposes the fitted frequencies to $qubit.ext.reset_f1 and "
            "$qubit.ext.reset_f2 only. DualTonePower proposes the tested "
            "reset module after analysing gains; length analysis has no "
            "automatic writeback."
        ),
        recommended=(
            "Keep each axis a tight span (a few MHz) around the predicted "
            "sideband; the 2D scan cost grows with both axes. Analysis denoises "
            "the map before peak finding; wavelet smoothing is the default and "
            "Gaussian remains available for comparison."
        ),
    )

    def run(
        self,
        req: RunRequest,
        cfg: FreqCfg,
        *,
        context: RunContext,
    ) -> DualToneFreqRunResult:
        del req
        cfg = cfg.with_updates(method="hard")
        result = FreqExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self,
        req: AnalyzeRequest[DualToneFreqRunResult, DualToneFreqAnalyzeParams],
        *,
        plots: Plots,
    ) -> DualToneFreqAnalyzeResult:
        params = req.analyze_params
        result = FreqExp().analyze(
            req.run_result,
            FreqAnalyzeOptions(
                smooth=params.smooth, smooth_method=params.smooth_method
            ),
            plots=plots,
        )
        return DualToneFreqAnalyzeResult(freq1=result.freq1, freq2=result.freq2)

    def get_writeback_items(
        self,
        req: WritebackRequest[DualToneFreqRunResult, DualToneFreqAnalyzeResult],
    ) -> Sequence[WritebackItem]:
        """Propose this analysis's two frequencies to captured qubit ext."""
        result = req.analyze_result
        items: list[WritebackItem] = [
            ParameterWriteback(
                target_name="$qubit.ext.reset_f1",
                description="Dual-tone reset frequency 1 (MHz)",
                proposed_value=result.freq1,
            ),
            ParameterWriteback(
                target_name="$qubit.ext.reset_f2",
                description="Dual-tone reset frequency 2 (MHz)",
                proposed_value=result.freq2,
            ),
        ]
        return items

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_dualreset_freq_{time.strftime('%m%d')}"
