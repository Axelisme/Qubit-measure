from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, TypeAlias, cast

from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    AnalyzeRequest,
    AnalyzeResultBase,
    ParameterWriteback,
    ParamMeta,
    PostAnalyzeRequest,
    PostAnalyzeResultBase,
    PostWritebackRequest,
    RunRequest,
    SessionEnv,
    T_PostAnalyzeResult,
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
    scaled_path_seed,
)
from zcu_tools.gui.app.measure.spec_helpers import (
    make_pulse_module_spec,
    make_pulse_readout_module_spec,
    make_reset_module_spec,
)
from zcu_tools.gui.cfg import (
    ReferenceValue,
    make_custom_reference_key,
    make_default_value,
)

from zcu_lab.v2._support.measure.analyze_results import fit_quality_summary
from zcu_lab.v2._support.measure.defaults.role_factories import ROLE_FACTORIES
from zcu_lab.v2.singleshot.ge.core import (
    GE_Cfg,
    GE_Exp,
    GE_Result,
    GEAnalysis,
    GEAnalyzeOptions,
    GEPostAnalyzeOptions,
)

if TYPE_CHECKING:
    from zcu_tools.plotting.plots import Plots

GERunResult: TypeAlias = RunRecord[GE_Cfg, GE_Result]


def _blank_pulse(ctx: SessionEnv) -> ReferenceValue:
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
    value.with_field(
        "ro_cfg.trig_offset",
        path_seed(
            "$resonator.ext.time_of_fly",
            expr="$resonator.ext.time_of_fly + 0.05",
            fallback=0.55,
        ).resolve(ctx),
    )
    return ReferenceValue(make_custom_reference_key("Pulse Readout"), value)


@dataclass
class GEAnalyzeParams:
    initial_state: Annotated[
        Literal["ground", "excited"], ParamMeta(label="Initial State")
    ] = "ground"
    # ``backend`` selects the primary rotation/threshold fit. Post-analysis uses
    # the resulting centres and does not choose or run another fit backend.
    backend: Annotated[Literal["pca", "center"], ParamMeta(label="Backend")] = "pca"
    logscale: Annotated[bool, ParamMeta(label="Log Scale")] = False
    align_t1: Annotated[bool, ParamMeta(label="Align T1")] = True
    length_ratio: Annotated[
        float | None, ParamMeta(label="Length Ratio", decimals=4)
    ] = None


@dataclass(frozen=True)
class GEAnalyzeResult(GEAnalysis, AnalyzeResultBase):
    """Fit calibration, with a JSON-safe GUI projection of the core result."""

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "initial_state": self.initial_state,
            "fidelity": self.fidelity,
            "theta": self.theta,
            "threshold": self.threshold,
            "ge_s": self.ge_s,
            "init_pops": self.init_pops.tolist(),
            "fit_quality": fit_quality_summary(self.fit_quality),
        }


@dataclass
class GEPostAnalyzeParams:
    """The GE confusion diagnostic has no independent operator parameters."""


@dataclass(frozen=True)
class GEPostAnalyzeResult(PostAnalyzeResultBase):
    ge_radius: float
    confusion: list[list[float]]


class GEAdapter(BaseAdapter[GE_Cfg, GERunResult, GEAnalyzeResult, GEAnalyzeParams]):
    experiment_tag: ClassVar[str | None] = GE_Exp.persistence_tag()
    exp_cls = GE_Exp
    ExpCfg_cls: ClassVar[Any] = GE_Cfg
    # FIT primary analysis + a confusion-diagnostic post-analysis layer.
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.FIT, post_analysis=True, load_data=True
    )

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        """Declare static shape and fresh defaults on the supplied capture.

        Instantiate with SessionEnv.parameters. Bound missing/null scalar
        calibration uses the guide's fallback. Missing module slots disable
        optional init/reset and seed required probe/readout inline. Unbound
        roles remain cfg Invalid; source/materialization faults propagate.
        """
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
                "modules.init_pulse",
                spec=make_pulse_module_spec(optional=True),
                default=module_seed("$qubit.module.x180", fallback=None),
            )
            .field(
                "modules.probe_pulse",
                spec=make_pulse_module_spec(label="Probe Pulse"),
                default=module_seed(
                    "$qubit.module.x180",
                    fallback=custom(_blank_pulse, description="inline probe pulse"),
                ),
            )
            .field(
                "modules.readout",
                spec=make_pulse_readout_module_spec(),
                default=module_seed(
                    "$qubit.module.readout",
                    fallback=custom(_blank_readout, description="inline readout"),
                ),
            )
            .relax_delay(
                scaled_path_seed("$qubit.t1", factor=5.0, fallback_value=100.0)
            )
            .int("shots", label="Shots", default=100000)
            .reps(1, locked=True)
            .rounds(1, locked=True)
            .build()
        )

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> GERunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        result = GE_Exp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self, req: AnalyzeRequest[GERunResult, GEAnalyzeParams], *, plots: Plots
    ) -> GEAnalyzeResult:
        params = req.analyze_params
        analysis = GE_Exp().analyze(
            req.run_result,
            GEAnalyzeOptions(
                initial_state=params.initial_state,
                backend=params.backend,
                logscale=params.logscale,
                align_t1=params.align_t1,
                length_ratio=params.length_ratio,
            ),
            plots=plots,
        )
        return GEAnalyzeResult(
            initial_state=analysis.initial_state,
            fidelity=analysis.fidelity,
            theta=analysis.theta,
            threshold=analysis.threshold,
            ge_s=analysis.ge_s,
            g_center=analysis.g_center,
            e_center=analysis.e_center,
            init_pops=analysis.init_pops,
            fit_quality=analysis.fit_quality,
        )

    def get_post_analyze_params(
        self, analyze_result: GEAnalyzeResult, ctx: SessionEnv
    ) -> GEPostAnalyzeParams:
        del analyze_result, ctx
        return GEPostAnalyzeParams()

    def post_analyze(
        self,
        req: PostAnalyzeRequest[GERunResult, GEAnalyzeResult, GEPostAnalyzeParams],
        *,
        plots: Plots,
    ) -> GEPostAnalyzeResult:
        analysis = GE_Exp().post_analyze(
            req.run_result, req.analyze_result, GEPostAnalyzeOptions(), plots=plots
        )
        return GEPostAnalyzeResult(
            ge_radius=analysis.confusion.radius,
            confusion=analysis.confusion.matrix.tolist(),
        )

    def get_writeback_items(
        self, req: WritebackRequest[GERunResult, GEAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        result.validate_calibration()
        # AcceptedWrite owns native complex conversion; producers keep numbers.
        return [
            ParameterWriteback(
                target_name="$qubit.ext.readout_cal.fidelity",
                description="Single-shot assignment fidelity",
                proposed_value=result.fidelity,
            ),
            ParameterWriteback(
                target_name="$qubit.ext.readout_cal.sigma",
                description="Single-shot IQ cluster width (s)",
                proposed_value=result.ge_s,
            ),
            ParameterWriteback(
                target_name="$qubit.ext.readout_cal.g_center",
                description="Single-shot |g> IQ cluster centre (complex)",
                proposed_value=result.g_center,
            ),
            ParameterWriteback(
                target_name="$qubit.ext.readout_cal.e_center",
                description="Single-shot |e> IQ cluster centre (complex)",
                proposed_value=result.e_center,
            ),
        ]

    def get_post_writeback_items(
        self,
        req: PostWritebackRequest[GERunResult, GEAnalyzeResult, T_PostAnalyzeResult],
    ) -> Sequence[WritebackItem]:
        result = cast(GEPostAnalyzeResult, req.post_analyze_result)
        return [
            ParameterWriteback(
                target_name="$qubit.ext.readout_cal.radius",
                description="Single-shot classification radius",
                proposed_value=result.ge_radius,
            ),
            ParameterWriteback(
                target_name="$qubit.ext.readout_cal.confusion_matrix",
                description="Single-shot 3x3 confusion matrix (prepared->measured)",
                proposed_value=result.confusion,
            ),
        ]

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_sh_ge_{time.strftime('%m%d')}"
