from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, TypeAlias

from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    AnalyzeResultBase,
    NoAnalyzeParams,
    RunRequest,
    SessionEnv,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.app.measure.schema_builder import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
)
from zcu_tools.gui.app.measure.seeds import SweepDefault, scaled_path_seed
from zcu_tools.gui.cfg import (
    EvalValue,
    ScalarSpec,
)
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.defaults.role_factories import ROLE_FACTORIES
from zcu_lab.v2._support.measure.singleshot_shared import read_confusion_matrix
from zcu_lab.v2.singleshot.t1.t1.core import T1AnalyzeOptions, T1Cfg, T1Exp, T1Result

# Transition-rate analysis publishes figures without numeric writeback.
SsT1RunResult: TypeAlias = RunRecord[T1Cfg, T1Result]


@dataclass
class SsT1AnalyzeResult(AnalyzeResultBase):
    pass


class SsT1Adapter(
    BaseAdapter[T1Cfg, SsT1RunResult, SsT1AnalyzeResult, NoAnalyzeParams]
):
    experiment_tag: ClassVar[str | None] = T1Exp.persistence_tag()
    exp_cls = T1Exp
    ExpCfg_cls: ClassVar[Any] = T1Cfg

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder(
                role_factories=ROLE_FACTORIES,
            )
            .reset(optional=True)
            .pulse("pi_pulse", role_id="pi_pulse")
            .readout()
            .relax_delay(
                scaled_path_seed("$qubit.t1", factor=5.0, fallback_value=100.0)
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
            .bool("uniform", label="Uniform (linear) sweep", default=False)
            .reps(1000)
            .rounds(10)
            .build()
        )

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> SsT1RunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        return RunRecord(cfg, T1Exp().run(cfg, context=context))

    def analyze(
        self, req: AnalyzeRequest[SsT1RunResult, NoAnalyzeParams], *, plots: Plots
    ) -> SsT1AnalyzeResult:
        # ``confusion_matrix`` is the GE 3×3 readout-correction matrix from md.
        # ``skip`` is not exposed as a user knob — users can re-run with a shorter
        # sweep instead.
        confusion = read_confusion_matrix(req.parameters)
        T1Exp().analyze(
            req.run_result, T1AnalyzeOptions(confusion_matrix=confusion), plots=plots
        )
        return SsT1AnalyzeResult()

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_ss_t1_{time.strftime('%m%d')}"
