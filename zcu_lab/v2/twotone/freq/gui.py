from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, TypeAlias

from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
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
from zcu_tools.gui.app.measure.schema_builder import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
)
from zcu_tools.gui.app.measure.seeds import (
    custom,
    module_seed,
    path_seed,
)
from zcu_tools.gui.app.measure.spec_helpers import (
    make_pulse_module_spec,
    make_pulse_readout_module_spec,
    make_readout_module_spec,
    make_reset_module_spec,
)
from zcu_tools.gui.cfg import (
    ReferenceValue,
    make_custom_reference_key,
    make_default_value,
)
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.defaults.role_factories import ROLE_FACTORIES
from zcu_lab.v2._support.measure.seeds import (
    qub_freq_range,
)
from zcu_lab.v2.twotone.freq.core import (
    FreqAnalyzeOptions,
    FreqCfg,
    FreqExp,
    FreqResult,
)

FreqRunResult: TypeAlias = RunRecord[FreqCfg, FreqResult]


def _blank_probe(ctx: SessionEnv) -> ReferenceValue:
    value = make_default_value(make_pulse_module_spec().allowed[0])
    value.with_field("freq", 0.0)
    value.with_field("ch", path_seed("$qubit.wiring.L01.ch", fallback=0).resolve(ctx))
    value.with_field("nqz", 2)
    value.with_field("gain", 0.1)
    value.with_field("waveform.length", 5.0)
    return ReferenceValue(make_custom_reference_key("Pulse"), value)


def _blank_readout(ctx: SessionEnv) -> ReferenceValue:
    value = make_default_value(make_pulse_readout_module_spec().allowed[0])
    frequency = path_seed("$resonator.freq", fallback=6000.0).resolve(ctx)
    value.with_field("pulse_cfg.freq", frequency)
    value.with_field(
        "pulse_cfg.ch", path_seed("$resonator.wiring.ch", fallback=0).resolve(ctx)
    )
    value.with_field("pulse_cfg.nqz", 2)
    value.with_field("pulse_cfg.gain", 0.1)
    value.with_field("pulse_cfg.waveform.length", 1.0)
    value.with_field("ro_cfg.ro_freq", frequency)
    value.with_field(
        "ro_cfg.ro_ch", path_seed("$resonator.wiring.ro_ch", fallback=0).resolve(ctx)
    )
    value.with_field("ro_cfg.ro_length", 0.9)
    value.with_field("ro_cfg.trig_offset", 0.55)
    return ReferenceValue(make_custom_reference_key("Pulse Readout"), value)


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

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

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

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder(
                role_factories=ROLE_FACTORIES,
            )
            .field(
                "modules.reset",
                spec=make_reset_module_spec(optional=True),
                default=module_seed("$qubit.module.reset", fallback=None),
            )
            .field(
                "modules.qub_pulse",
                spec=make_pulse_module_spec(label="Probe Pulse").lock_literal(
                    "freq", 0.0
                ),
                default=custom(_blank_probe, description="inline probe pulse"),
            )
            .field(
                "modules.readout",
                spec=make_readout_module_spec(),
                default=module_seed(
                    "$qubit.module.readout",
                    fallback=custom(_blank_readout, description="inline readout"),
                ),
            )
            .relax_delay(1.0)
            .sweep(
                "freq",
                label="Freq (MHz)",
                default=qub_freq_range(expts=301),
            )
            .reps(1000)
            .rounds(100)
            .build()
        )

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
