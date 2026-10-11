from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, TypeAlias

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
from zcu_tools.gui.cfg import (
    ReferenceValue,
    make_custom_reference_key,
    make_default_value,
)
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.schema_builder import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
)
from zcu_lab.v2._support.measure.seeds import custom, path_seed
from zcu_lab.v2._support.measure.spec_helpers import (
    make_pulse_module_spec,
    make_pulse_readout_module_spec,
    make_reset_module_spec,
)
from zcu_lab.v2.lookback.core import (
    LookbackAnalyzeOptions,
    LookbackCfg,
    LookbackExp,
    LookbackResult,
)

LookbackRunResult: TypeAlias = RunRecord[LookbackCfg, LookbackResult]


def _blank_readout(ctx: SessionEnv) -> ReferenceValue:
    """Seed a fresh inline pulse-readout from the fixed resonator capture.

    Frequency defaults to 6000 MHz, wiring channels to zero, and trigger offset
    to 0.4 us when calibration is absent/null. Bound calibration stays expressed
    on the capture; unbound roles and source faults do not become fallback.
    """
    spec = make_pulse_readout_module_spec().allowed[0]
    frequency = path_seed("$resonator.freq", fallback=6000.0).resolve(ctx)
    return (
        ReferenceValue(
            make_custom_reference_key(spec.label or "Readout"),
            make_default_value(spec),
        )
        .with_field("pulse_cfg.freq", frequency)
        .with_field("pulse_cfg.gain", 1.0)
        .with_field("pulse_cfg.nqz", 2)
        .with_field("pulse_cfg.waveform.length", 1.0)
        .with_field(
            "pulse_cfg.ch",
            path_seed("$resonator.wiring.ch", fallback=0).resolve(ctx),
        )
        .with_field("ro_cfg.ro_freq", frequency)
        .with_field(
            "ro_cfg.ro_ch",
            path_seed("$resonator.wiring.ro_ch", fallback=0).resolve(ctx),
        )
        .with_field("ro_cfg.ro_length", 1.5)
        .with_field(
            "ro_cfg.trig_offset",
            path_seed(
                "$resonator.ext.time_of_fly",
                expr="$resonator.ext.time_of_fly - 0.1",
                fallback=0.4,
            ).resolve(ctx),
        )
    )


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
            "Defaults use the tab's fixed qubit and referenced resonator. "
            "Resonator frequency seeds the readout pulse and ADC at once; "
            "resonator wiring supplies drive and ADC channels. Trigger offset "
            "is resonator.ext.time_of_fly - 0.1 us, with fallback 0.4 us. "
            "Missing or null calibration uses fallback; an unbound resonator "
            "leaves cfg Invalid until supplied."
        ),
        expects_ml=(
            "Readout always starts as a fresh inline pulse-readout, even "
            "when a calibrated readout slot exists. It does not adopt a "
            "legacy ro_waveform. Optional reset and init pulse start disabled; "
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

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        """Declare static shape and fresh defaults on SessionEnv.parameters.

        Reset and init start disabled. Readout always starts inline on the fixed
        resonator capture, with trigger offset time_of_fly minus 0.1 us or 0.4 us
        for absent/null calibration. An unbound resonator leaves cfg Invalid;
        unused qubit binding is not required. Source and materialization faults
        propagate without fallback.
        """
        return (
            MeasureCfgBuilder()
            .field(
                "modules.reset",
                spec=make_reset_module_spec(optional=True),
                default=None,
            )
            .field(
                "modules.init_pulse",
                spec=make_pulse_module_spec(optional=True),
                default=None,
            )
            .field(
                "modules.readout",
                spec=make_pulse_readout_module_spec(),
                default=custom(_blank_readout, description="inline lookback readout"),
            )
            .relax_delay(0.0)
            .reps(1, locked=True)
            .rounds(500)
            .build()
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
        return f"lookback_{time.strftime('%H%M')}"
