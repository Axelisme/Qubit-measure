from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, TypeAlias

from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    MetaDictWriteback,
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
from zcu_tools.resources.entry import RoleSpec

from zcu_lab.v2._support.measure.analyze_results import fit_quality_summary
from zcu_lab.v2._support.measure.schema_builder import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
)
from zcu_lab.v2._support.measure.seeds import (
    SweepDefault,
    custom,
    module_seed,
    path_seed,
    scaled_path_seed,
)
from zcu_lab.v2._support.measure.spec_helpers import (
    make_pulse_module_spec,
    make_pulse_readout_module_spec,
    make_readout_module_spec,
    make_reset_module_spec,
)
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
    value.with_field("freq", path_seed("qubit.freq", fallback=4000.0).resolve(ctx))
    value.with_field("ch", path_seed("qubit.wiring.L01.ch", fallback=0).resolve(ctx))
    value.with_field("nqz", 2)
    value.with_field("gain", 0.1)
    value.with_field("waveform.length", 5.1)
    return ReferenceValue(make_custom_reference_key("Pulse"), value)


def _blank_readout(ctx: SessionEnv) -> ReferenceValue:
    value = make_default_value(make_pulse_readout_module_spec().allowed[0])
    frequency = path_seed("resonator.freq", fallback=6000.0).resolve(ctx)
    value.with_field("pulse_cfg.freq", frequency)
    value.with_field(
        "pulse_cfg.ch", path_seed("resonator.wiring.ch", fallback=0).resolve(ctx)
    )
    value.with_field("pulse_cfg.nqz", 2)
    value.with_field("pulse_cfg.gain", 0.1)
    value.with_field("pulse_cfg.waveform.length", 1.0)
    value.with_field("ro_cfg.ro_freq", frequency)
    value.with_field(
        "ro_cfg.ro_ch", path_seed("resonator.wiring.ro_ch", fallback=0).resolve(ctx)
    )
    value.with_field("ro_cfg.ro_length", 0.9)
    value.with_field("ro_cfg.trig_offset", 0.55)
    return ReferenceValue(make_custom_reference_key("Pulse Readout"), value)


class T1Adapter(BaseAdapter[T1Cfg, T1RunResult, T1AnalyzeResult, T1AnalyzeParams]):
    exp_cls = T1Exp
    experiment_tag: ClassVar[str | None] = T1Exp().persistence_spec().tag
    ExpCfg_cls: ClassVar[Any] = T1Cfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "T1 energy relaxation: applies a pi pulse to excite the qubit, "
            "then sweeps a wait delay before readout and fits the exponential "
            "decay to extract T1. Runs on real hardware. Run after a pi pulse "
            "has been calibrated (amplitude/length Rabi)."
        ),
        expects_md=(
            "Requires a qubit and its referenced resonator. Defaults read "
            "qubit.t1 (us), qubit.freq (MHz), qubit.wiring.L01.ch, "
            "resonator.freq (MHz), and resonator.wiring.ch/ro_ch from the "
            "tab's fixed capture. Missing T1 seeds relax_delay at 100 us "
            "and the delay sweep stop at 500 us."
        ),
        expects_ml=(
            "Uses qubit.module.x180 and qubit.module.readout from the fixed "
            "capture. Unset or dangling slots seed inline pulse/readout "
            "defaults. Optional qubit.module.reset stays disabled when absent. "
            "Inline readout trigger offset defaults to 0.55 us; verify wiring "
            "and timing before running."
        ),
        typical_writeback=(
            "Proposes fitted qubit.t1 and its standard error (us) to the "
            "captured qubit and working point. No module writeback."
        ),
        recommended=(
            "Analysis defaults to a single-exponential fit; enable "
            "dual-exponential only when the decay clearly shows two timescales "
            "(a fast component on top of the slow relaxation). A delay sweep "
            "reaching ~5*T1 lets the decay flatten so the fit is "
            "well-constrained; with no prior qubit.t1 it spans 0–500 us. "
            "Set 'uniform=False' to cluster more points on the exponential "
            "decay while keeping the configured start/stop window."
        ),
    )

    def parameter_roles(self) -> Mapping[str, RoleSpec]:
        """Capture the qubit and its referenced resonator when opening a tab."""
        return {
            "qubit": RoleSpec("qubit/*"),
            "resonator": RoleSpec("resonator", via="qubit.resonator"),
        }

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder()
            .field(
                "modules.reset",
                spec=make_reset_module_spec(optional=True),
                default=module_seed("qubit.module.reset", fallback=None),
            )
            .field(
                "modules.pi_pulse",
                spec=make_pulse_module_spec(),
                default=module_seed(
                    "qubit.module.x180",
                    fallback=custom(_blank_pi_pulse, description="inline pi pulse"),
                ),
            )
            .field(
                "modules.readout",
                spec=make_readout_module_spec(),
                default=module_seed(
                    "qubit.module.readout",
                    fallback=custom(_blank_readout, description="inline readout"),
                ),
            )
            .relax_delay(
                scaled_path_seed("qubit.t1", factor=5.0, fallback_value=100.0),
            )
            .sweep(
                "length",
                label="Delay (us)",
                default=SweepDefault(
                    start=0.0,
                    # The fallback is the fully scaled 5 * 100 us window.
                    stop=scaled_path_seed("qubit.t1", factor=5.0, fallback_value=500.0),
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
        result = req.analyze_result
        return [
            MetaDictWriteback(
                target_name="qubit.t1",
                description="T1 relaxation time (us)",
                proposed_value=result.t1,
                stderr=result.t1_err,
            ),
        ]

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_t1_{time.strftime('%m%d')}"
