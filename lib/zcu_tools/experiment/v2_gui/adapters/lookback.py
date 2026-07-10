from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, TypeAlias

from matplotlib.figure import Figure

from zcu_tools.experiment.v2.lookback import LookbackCfg, LookbackExp, LookbackResult
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    CfgBuilder,
    RoleInit,
    build_exp_spec,
    make_pulse_module_spec,
    make_pulse_readout_module_spec,
    make_reset_module_spec,
    make_trig_offset,
)
from zcu_tools.gui.app.main.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    ExpContext,
    MetaDictWriteback,
    ParamMeta,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.cfg import (
    CfgSectionSpec,
    CfgSectionValue,
    LiteralSpec,
)

LookbackRunResult: TypeAlias = LookbackResult


@dataclass
class LookbackAnalyzeParams:
    ratio: Annotated[float, ParamMeta(label="Threshold ratio", decimals=3)] = 0.1
    smooth: Annotated[float, ParamMeta(label="Smooth sigma", decimals=3)] = 1.0
    plot_fit: Annotated[bool, ParamMeta(label="Plot fit")] = True


@dataclass
class LookbackAnalyzeResult(AnalyzeResultBase):
    predict_offset: float
    figure: Figure


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
    def cfg_spec(cls) -> CfgSectionSpec:
        return build_exp_spec(
            modules={
                "reset": make_reset_module_spec(optional=True),
                "init_pulse": make_pulse_module_spec(optional=True),
                "readout": make_pulse_readout_module_spec(),
            },
            # reps is locked to 1: decimated acquisition has no multi-rep
            # averaging, so LookbackExp.run() forces reps=1 anyway. Locking it
            # here means the GUI never offers a value that gets silently
            # overridden.
            reps=LiteralSpec(value=1, label="Reps"),
        )

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        # reps is locked to 1 by the LiteralSpec in cfg_spec(); the L1 blank
        # already carries that literal, so it is not set here. The optional
        # init_pulse / reset default to disabled (None) via the L1 blank too.
        return (
            CfgBuilder(ctx, self.cfg_spec())
            .scalars(rounds=500, relax_delay=0.0)
            .role("modules.readout", "readout", RoleInit.INLINE)
            .set("modules.readout.pulse_cfg.gain", 1.0)
            .set("modules.readout.ro_cfg.ro_length", 1.5)
            .set(
                "modules.readout.ro_cfg.trig_offset",
                make_trig_offset(ctx, trig_expr="timeFly - 0.1", trig_fallback=0.4),
            )
            .build()
        )

    def analyze(
        self,
        req: AnalyzeRequest[LookbackRunResult, LookbackAnalyzeParams],
    ) -> LookbackAnalyzeResult:
        params = req.analyze_params
        offset, figure = LookbackExp().analyze(
            req.run_result,
            ratio=params.ratio,
            smooth=params.smooth,
            plot_fit=params.plot_fit,
        )
        return LookbackAnalyzeResult(predict_offset=offset, figure=figure)

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

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"lookback_{time.strftime('%H%M')}"
