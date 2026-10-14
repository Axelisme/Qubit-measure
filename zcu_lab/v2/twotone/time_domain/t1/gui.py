from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, TypeAlias

from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
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
    SweepDefault,
    custom,
    module_seed,
    path_seed,
    scaled_path_seed,
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

from zcu_lab.v2._support.measure.analyze_results import fit_quality_summary
from zcu_lab.v2._support.measure.defaults.role_factories import ROLE_FACTORIES
from zcu_lab.v2.twotone.time_domain.t1.core import (
    T1Analysis,
    T1AnalyzeOptions,
    T1Cfg,
    T1Exp,
    T1Result,
)

if TYPE_CHECKING:
    from zcu_tools.plotting.plots import Plots

T1RunResult: TypeAlias = RunRecord[T1Cfg, T1Result]


@dataclass
class T1AnalyzeParams:
    dual_exp: Annotated[bool, ParamMeta(label="Dual exponential")] = False
    skip: Annotated[int, ParamMeta(label="Skip leading points")] = 0


@dataclass(frozen=True)
class T1AnalyzeResult:
    """GUI summary over the unchanged, typed core analysis."""

    analysis: T1Analysis

    @property
    def t1(self) -> float:
        return self.analysis.t1

    @property
    def t1_err(self) -> float:
        return self.analysis.t1_err

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "t1": self.analysis.t1,
            "t1_err": self.analysis.t1_err,
            "t1b": self.analysis.t1b,
            "t1b_err": self.analysis.t1b_err,
            "fit_quality": fit_quality_summary(self.analysis.fit_quality),
        }


def _blank_pi_pulse(ctx: SessionEnv) -> ReferenceValue:
    value = make_default_value(make_pulse_module_spec().allowed[0])
    value.with_field("freq", path_seed("$qubit.freq", fallback=4000.0).resolve(ctx))
    value.with_field("ch", path_seed("$qubit.wiring.L01.ch", fallback=0).resolve(ctx))
    value.with_field("nqz", 2)
    value.with_field("gain", 0.1)
    value.with_field("waveform.length", 5.1)
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


class T1Adapter(BaseAdapter[T1Cfg, T1RunResult, T1AnalyzeResult, T1AnalyzeParams]):
    exp_cls = T1Exp
    experiment_tag: ClassVar[str | None] = T1Exp.persistence_tag()
    ExpCfg_cls: ClassVar[Any] = T1Cfg

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

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
                "modules.pi_pulse",
                spec=make_pulse_module_spec(),
                default=module_seed(
                    "$qubit.module.x180",
                    fallback=custom(_blank_pi_pulse, description="inline pi pulse"),
                ),
            )
            .field(
                "modules.readout",
                spec=make_readout_module_spec(),
                default=module_seed(
                    "$qubit.module.readout",
                    fallback=custom(_blank_readout, description="inline readout"),
                ),
            )
            .relax_delay(
                scaled_path_seed("$qubit.t1", factor=5.0, fallback_value=100.0),
            )
            .sweep(
                "length",
                label="Delay (us)",
                default=SweepDefault(
                    start=0.0,
                    # The fallback is the fully scaled 5 * 100 us window.
                    stop=scaled_path_seed(
                        "$qubit.t1", factor=5.0, fallback_value=500.0
                    ),
                    expts=101,
                ),
            )
            .bool(
                "uniform",
                label="Uniform (linear) sweep",
                default=True,
            )
            .reps(1000)
            .rounds(100)
            .build()
        )

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> T1RunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        result = T1Exp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self, req: AnalyzeRequest[T1RunResult, T1AnalyzeParams], *, plots: Plots
    ) -> T1AnalyzeResult:
        params = req.analyze_params
        if params.skip < 0:
            raise ValueError("Skip leading points must be nonnegative")
        analysis = T1Exp().analyze(
            req.run_result,
            T1AnalyzeOptions(dual_exp=params.dual_exp, skip=params.skip),
            plots=plots,
        )
        return T1AnalyzeResult(analysis)

    def get_writeback_items(
        self, req: WritebackRequest[T1RunResult, T1AnalyzeResult]
    ) -> Sequence[WritebackItem]:
        """Propose fitted T1 parameters and a captured-component sample.

        req supplies the source result and fixed Run context. Sample fields use
        us and target the Run's resolved $qubit, never the live selection. Without
        a parameter capture, retain parameter proposals only so loaded data can
        still be analyzed; Apply rejects the missing capture. Unbound $qubit
        with a capture raises RoleResolutionError. WritebackService supplies
        bias coordinates or omits the sample with its existing public reason.
        No write or device access occurs here; this Run leaves devices fixed.
        """
        result = req.analyze_result
        items: list[WritebackItem] = [
            ParameterWriteback(
                target_name="$qubit.t1",
                description="T1 relaxation time (us)",
                proposed_value=result.t1,
                stderr=result.t1_err,
            ),
        ]
        parameters = req.ctx.parameters
        if parameters is not None:
            items.append(
                SampleWriteback(
                    target_name=parameters.resolve_path("$qubit"),
                    description="T1 sample at the captured bias",
                    proposed_value={
                        "t1_us": result.t1,
                        "t1_err_us": result.t1_err,
                    },
                )
            )
        return items

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_t1_{time.strftime('%m%d')}"
