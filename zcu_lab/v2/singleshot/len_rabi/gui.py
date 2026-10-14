from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, TypeAlias

from zcu_tools.analysis.fitting.singleshot_rabi import RabiJointFitResult
from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    AnalyzeResultBase,
    ParamMeta,
    RunRequest,
    SessionEnv,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.app.measure.schema_builder import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    ModuleInit,
)
from zcu_tools.gui.app.measure.seeds import SweepDefault, scaled_path_seed
from zcu_tools.gui.cfg import (
    EvalValue,
    ScalarSpec,
)
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.defaults.role_factories import ROLE_FACTORIES
from zcu_lab.v2._support.measure.singleshot_rabi import rabi_calibration_writeback
from zcu_lab.v2.singleshot.len_rabi.core import (
    LenRabiAnalyzeOptions,
    LenRabiCfg,
    LenRabiExp,
    LenRabiResult,
)

# ``LenRabiExp`` from ``singleshot`` — sweeps the qubit-drive pulse *length* and
# preserves every raw IQ shot. Analysis derives populations from that canonical
# raw result rather than persisting a second population representation.
SsLenRabiRunResult: TypeAlias = RunRecord[LenRabiCfg, LenRabiResult]


@dataclass
class SsLenRabiAnalyzeParams:
    initial_state: Annotated[
        Literal["ground", "excited"], ParamMeta(label="Initial State")
    ] = "ground"
    decay: Annotated[bool, ParamMeta(label="Fit decay envelope")] = True
    fit_phase: Annotated[bool, ParamMeta(label="Fit phase offset")] = False


@dataclass
class SsLenRabiAnalyzeResult(AnalyzeResultBase):
    # The full numeric fit is intentionally non-JSON-safe and therefore omitted
    # from the GUI summary. The operator reviews the population/fit Figure while
    # writeback projection reads the typed domain result directly.
    fit_result: RabiJointFitResult


class SsLenRabiAdapter(
    BaseAdapter[
        LenRabiCfg, SsLenRabiRunResult, SsLenRabiAnalyzeResult, SsLenRabiAnalyzeParams
    ]
):
    experiment_tag: ClassVar[str | None] = LenRabiExp.persistence_tag()
    exp_cls = LenRabiExp
    ExpCfg_cls: ClassVar[Any] = LenRabiCfg

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder(
                role_factories=ROLE_FACTORIES,
            )
            .reset(optional=True)
            .pulse(
                "qub_pulse",
                role_id="qub_probe",
                init=ModuleInit.INLINE,
                overrides={"gain": 1.0},
            )
            .readout()
            .relax_delay(50.5)
            .sweep(
                "length",
                label="Length (us)",
                default=SweepDefault(
                    start=0.03,
                    stop=scaled_path_seed(
                        "$qubit.module.x180.waveform.length",
                        factor=4.0,
                        fallback_value=0.2,
                    ),
                    expts=51,
                ),
            )
            .int("shots", label="Shots", default=1000)
            .field(
                "g_center",
                spec=ScalarSpec("Ground center", complex),
                default=EvalValue("$qubit.ext.readout_cal.g_center"),
            )
            .field(
                "e_center",
                spec=ScalarSpec("Excited center", complex),
                default=EvalValue("$qubit.ext.readout_cal.e_center"),
            )
            .field(
                "radius",
                spec=ScalarSpec("Classification radius", float),
                default=EvalValue("$qubit.ext.readout_cal.radius"),
            )
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
    ) -> SsLenRabiRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        result = LenRabiExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self,
        req: AnalyzeRequest[SsLenRabiRunResult, SsLenRabiAnalyzeParams],
        *,
        plots: Plots,
    ) -> SsLenRabiAnalyzeResult:
        fit_result = LenRabiExp().analyze(
            req.run_result,
            LenRabiAnalyzeOptions(
                decay=req.analyze_params.decay,
                fit_phase=req.analyze_params.fit_phase,
                initial_state=req.analyze_params.initial_state,
            ),
            plots=plots,
        )
        return SsLenRabiAnalyzeResult(fit_result=fit_result)

    def get_writeback_items(
        self, req: WritebackRequest[SsLenRabiRunResult, SsLenRabiAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        return rabi_calibration_writeback(req.analyze_result.fit_result)

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_ss_len_rabi_{time.strftime('%m%d')}"
