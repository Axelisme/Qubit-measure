from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, ClassVar, TypeAlias

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
    SessionEnv,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.lookback.core import (
    LookbackAnalyzeOptions,
    LookbackCfg,
    LookbackExp,
    LookbackResult,
)
from zcu_lab.v2.lookback.form import FORM

LookbackRunResult: TypeAlias = RunRecord[LookbackCfg, LookbackResult]


@dataclass
class LookbackAnalyzeParams:
    ratio: Annotated[float, ParamMeta(label="Threshold ratio", decimals=3)] = 0.1
    smooth: Annotated[float, ParamMeta(label="Smooth sigma", decimals=3)] = 1.0
    plot_fit: Annotated[bool, ParamMeta(label="Plot fit")] = True


@dataclass
class LookbackAnalyzeResult(AnalyzeResultBase):
    predict_offset: float


class LookbackAdapter(
    BaseAdapter[
        LookbackCfg,
        LookbackRunResult,
        LookbackAnalyzeResult,
        LookbackAnalyzeParams,
    ]
):
    experiment_tag: ClassVar[str | None] = LookbackExp.persistence_tag()
    exp_cls = LookbackExp
    ExpCfg_cls = LookbackCfg
    form = FORM

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Readout time-of-flight (lookback) measurement: fires the readout "
            "pulse and captures the decimated ADC trace versus time, so you "
            "can see when the readout signal actually arrives at the digitizer. "
            "Runs on real hardware (reps is locked to 1 — decimated "
            "acquisition has no multi-rep averaging). Run this first when "
            "bringing up a new readout chain or after changing cabling, to "
            "calibrate the trigger offset."
        ),
        expects_md=(
            "Defaults use the tab's fixed resonator capture. resonator.freq "
            "seeds the pulse and ADC frequency; resonator.wiring.ch/ro_ch "
            "seed generator and ADC channels. Trigger offset is "
            "resonator.ext.time_of_fly - 0.1 us. Missing or null calibration "
            "and unbound aliases leave the affected fields required and cfg "
            "Invalid until supplied. No numeric fallback is substituted."
        ),
        expects_ml=(
            "Readout always starts as a fresh inline pulse-readout, even "
            "when a calibrated readout slot exists. Its const waveform is "
            "inline, not a named library waveform. Optional reset and init pulse start disabled; "
            "you can select their modules explicitly before Run."
        ),
        typical_writeback=(
            "Proposes the detected signal-arrival time in us at captured "
            "resonator.ext.time_of_fly. The writeback service resolves the "
            "destination on the analysis capture, not current focus. "
            "No module or waveform writeback."
        ),
        recommended=(
            "Analysis defaults: threshold 'ratio'=0.1 (the offset is the "
            "latest time before the peak where the magnitude is still below "
            "10% of maximum), 'smooth' sigma=1.0, plot-fit on. Use a wide "
            "readout window so the full pulse arrival is captured; raise "
            "'smooth' if the trace is noisy and the offset jitters, lower "
            "'ratio' if it triggers too early on baseline ripple."
        ),
    )

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> LookbackRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        result = LookbackExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self,
        req: AnalyzeRequest[LookbackRunResult, LookbackAnalyzeParams],
        *,
        plots: Plots,
    ) -> LookbackAnalyzeResult:
        params = req.analyze_params
        answer = LookbackExp().analyze(
            req.run_result,
            LookbackAnalyzeOptions(
                ratio=params.ratio,
                smooth=params.smooth,
                plot_fit=params.plot_fit,
            ),
            plots=plots,
        )
        return LookbackAnalyzeResult(predict_offset=answer.predict_offset)

    def get_writeback_items(
        self, req: WritebackRequest[LookbackRunResult, LookbackAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        """Propose detected arrival time in us at captured resonator.ext.time_of_fly.

        req supplies the analysis and its capture; source.cfg may be None.
        WritebackService resolves the destination on req.ctx.parameters and
        rejects missing capture or unbound roles. No live focus lookup or write
        occurs here.
        """
        return [
            ParameterWriteback(
                target_name="$resonator.ext.time_of_fly",
                description="Readout trigger offset prediction (us)",
                proposed_value=req.analyze_result.predict_offset,
            )
        ]

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        del ctx
        return f"lookback_{time.strftime('%H%M')}"
