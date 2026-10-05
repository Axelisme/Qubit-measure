from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    MetaDictWriteback,
    ParamMeta,
    RunRequest,
    SessionEnv,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.defaults.helpers import make_trig_offset
from zcu_lab.v2._support.measure.schema_builder import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    ModuleInit,
)
from zcu_lab.v2._support.measure.seeds import custom
from zcu_lab.v2.lookback.core import (
    LookbackAnalyzeOptions,
    LookbackCfg,
    LookbackExp,
    LookbackResult,
)

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
    exp_cls = LookbackExp
    ExpCfg_cls: ClassVar[Any] = LookbackCfg

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
            "Reads from the MetaDict (all optional): 'r_f' — resonator "
            "frequency, used for both the readout pulse and the ADC "
            "down-conversion (~4000–8000 MHz); 'res_ch' / 'ro_ch' — readout "
            "drive / ADC channel indices; 'timeFly' — a previously measured "
            "cable time-of-flight; when present the trigger offset is taken as "
            "'timeFly - 0.1' us, otherwise it falls back to ~0.4 us."
        ),
        expects_ml=(
            "Needs a pulse-readout module, and references a ModuleLibrary "
            "waveform named 'ro_waveform' for the readout pulse shape when one "
            "exists (optional). Optionally composes 'reset' / 'init_pulse' "
            "modules ahead of the readout."
        ),
        typical_writeback=(
            "Proposes the detected signal-arrival time back into MetaDict "
            "'timeFly' (us) — the seed for the trigger offset of every "
            "subsequent readout-based experiment. No ModuleLibrary writeback."
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

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder()
            .reset(optional=True, init=ModuleInit.DISABLED)
            .pulse(
                "init_pulse",
                role_id="pi_pulse",
                optional=True,
                init=ModuleInit.DISABLED,
            )
            .readout(
                pulse_only=True,
                init=ModuleInit.INLINE,
                overrides={
                    "pulse_cfg.gain": 1.0,
                    "ro_cfg.ro_length": 1.5,
                    "ro_cfg.trig_offset": custom(
                        lambda ctx: make_trig_offset(
                            ctx,
                            trig_expr="timeFly - 0.1",
                            trig_fallback=0.4,
                        ),
                        description="lookback trigger offset",
                    ),
                },
            )
            .relax_delay(0.0)
            .reps(1, locked=True)
            .rounds(500)
            .build()
        )

    def run(
        self, req: RunRequest, raw_cfg: dict[str, object], *, context: RunContext
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
        return [
            MetaDictWriteback(
                target_name="timeFly",
                description="Readout trigger offset prediction (us)",
                proposed_value=req.analyze_result.predict_offset,
            )
        ]

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"lookback_{time.strftime('%H%M')}"
