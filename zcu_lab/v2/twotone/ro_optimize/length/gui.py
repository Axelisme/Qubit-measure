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

from zcu_lab.v2._support.measure.writeback_helpers import readout_dpm_writeback_items
from zcu_lab.v2.twotone.ro_optimize.length.core import (
    LengthAnalyzeOptions,
    LengthCfg,
    LengthExp,
    LengthResult,
)
from zcu_lab.v2.twotone.ro_optimize.length.form import make_form

RoOptLengthRunResult: TypeAlias = RunRecord[LengthCfg, LengthResult]


@dataclass
class RoOptLengthAnalyzeParams:
    smooth_method: Annotated[
        Literal["wavelet", "gaussian"], ParamMeta(label="Smooth method")
    ] = "wavelet"
    smooth: Annotated[float, ParamMeta(label="Smooth strength", decimals=2)] = 1.0
    # GUI-facing name for LengthAnalyzeOptions.t0: this is a duration
    # normalization overhead, not a penalty strength. Small positive values
    # produce stronger short-readout bias than large positive values.
    duration_t0: Annotated[float | None, ParamMeta(label="Duration t0 (us)")] = None


@dataclass
class RoOptLengthAnalyzeResult(AnalyzeResultBase):
    best_length: float


class RoOptLengthAdapter(
    BaseAdapter[
        LengthCfg,
        RoOptLengthRunResult,
        RoOptLengthAnalyzeResult,
        RoOptLengthAnalyzeParams,
    ]
):
    experiment_tag: ClassVar[str | None] = LengthExp.persistence_tag()
    exp_cls = LengthExp

    @staticmethod
    def make_form(ctx: FormContext) -> Form[LengthCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Readout length optimization: with the qubit toggled between g and "
            "e by a pi pulse, sweeps the readout window length and measures the "
            "g/e signal-to-noise ratio (SNR), to pick the shortest readout that "
            "still resolves the states well. Runs on real hardware. A "
            "readout-tuning step, typically after the readout frequency and "
            "power are set."
        ),
        expects_md=(
            "Seeds 51 length points from 0.01 to 3.5 us. Relax delay reads 5.0 * "
            "$qubit.t1. Run replaces acquisition length and sets drive length from the "
            "largest requested length. Open and Reset evaluate starting expressions once "
            "into direct input using captured sources. Inline qubit pulses read "
            "$qubit.freq and $qubit.wiring.L01.ch. Inline readout reads $resonator.freq, "
            "$resonator.wiring.ch, $resonator.wiring.ro_ch and $resonator.ext.time_of_fly "
            "+ 0.05 us for trigger timing. Missing or invalid required sources leave the "
            "affected fields empty and Invalid; supply calibration or direct values "
            "before Run. No numeric calibration fallback is supplied."
        ),
        expects_ml=(
            "The qubit pulse uses $qubit.module.x180 as a captured library link, "
            "otherwise a fresh pi_pulse template with a const waveform. "
            "$qubit.module.readout selects a captured library link, otherwise a fresh "
            "inline readout template with a const waveform. Its required parameter "
            "sources must still be present. Optional $qubit.module.reset has no fallback: "
            "an unset/dangling slot or unbound alias leaves it disabled and Valid."
        ),
        typical_writeback=(
            "Proposes the SNR-maximizing readout length into parameter context "
            "$qubit.ext.best_ro_length (us). When a cfg snapshot with pulse readout is "
            "available and $qubit.ext.best_ro_freq / $qubit.ext.best_ro_gain / "
            "$qubit.ext.best_ro_length are known from this result plus parameter context, "
            "also proposes $qubit.readout.readout_dpm."
        ),
        recommended=(
            "Analysis denoises the SNR curve before picking the peak; wavelet "
            "smoothing is the default. The optional 'duration_t0' analyze param "
            "is the fixed-duration term in SNR/sqrt(length + t0): leave it "
            "blank (None) for the raw max; a small positive us value applies "
            "stronger short-readout bias than a large positive value."
        ),
    )

    def run(
        self,
        req: RunRequest,
        cfg: LengthCfg,
        *,
        context: RunContext,
    ) -> RoOptLengthRunResult:
        del req
        return RunRecord(cfg, LengthExp().run(cfg, context=context))

    def analyze(
        self,
        req: AnalyzeRequest[RoOptLengthRunResult, RoOptLengthAnalyzeParams],
        *,
        plots: Plots,
    ) -> RoOptLengthAnalyzeResult:
        params = req.analyze_params
        options = LengthAnalyzeOptions(
            t0=params.duration_t0,
            smooth=params.smooth,
            smooth_method=params.smooth_method,
        )
        result = LengthExp().analyze(req.run_result, options, plots=plots)
        return RoOptLengthAnalyzeResult(
            best_length=result.best_length,
        )

    def get_writeback_items(
        self, req: WritebackRequest[RoOptLengthRunResult, RoOptLengthAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        items: list[WritebackItem] = [
            ParameterWriteback(
                target_name="$qubit.ext.best_ro_length",
                description="Optimal readout length (us)",
                proposed_value=result.best_length,
            ),
        ]
        items.extend(
            readout_dpm_writeback_items(
                req.ctx,
                req.run_result.cfg,
                proposed={"best_ro_length": result.best_length},
            )
        )
        return items

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_ro_opt_length_{time.strftime('%m%d')}"
