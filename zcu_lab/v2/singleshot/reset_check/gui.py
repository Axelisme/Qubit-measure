from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

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
from zcu_tools.gui.cfg import EvalValue, ScalarSpec
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.defaults.role_factories import ROLE_FACTORIES
from zcu_lab.v2._support.measure.singleshot_shared import read_confusion_matrix
from zcu_lab.v2.singleshot.reset_check.core import (
    ResetCheckAnalyzeOptions,
    ResetCheckCfg,
    ResetCheckExp,
    ResetCheckResult,
)

SsResetCheckRunResult = RunRecord[ResetCheckCfg, ResetCheckResult]


@dataclass
class SsResetCheckAnalyzeResult(AnalyzeResultBase):
    reset_mean_excited_population: float
    reset_max_excited_population: float
    reset_max_other_population: float
    worst_sample_gain: float
    analyzed_reset_points: int


class SsResetCheckAdapter(
    BaseAdapter[
        ResetCheckCfg, SsResetCheckRunResult, SsResetCheckAnalyzeResult, NoAnalyzeParams
    ]
):
    experiment_tag: ClassVar[str | None] = ResetCheckExp.persistence_tag()
    exp_cls = ResetCheckExp
    ExpCfg_cls: ClassVar[Any] = ResetCheckCfg
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder(
                role_factories=ROLE_FACTORIES,
            )
            .reset(optional=True)
            .pulse("rabi_pulse", role_id="rabi_pulse", label="Rabi Pulse")
            .reset("tested_reset", role_id="reset", label="Tested Reset")
            .readout()
            .relax_delay(
                scaled_path_seed("$qubit.t1", factor=5.0, fallback_value=100.0)
            )
            .sweep(
                "gain",
                label="Gain (a.u.)",
                default=SweepDefault(
                    start=0.0,
                    stop=scaled_path_seed(
                        "$qubit.module.x180.gain", factor=4.0, fallback_value=1.0
                    ),
                    expts=51,
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
            .reps(5000)
            .rounds(1)
            .build()
        )

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> SsResetCheckRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        result = ResetCheckExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self,
        req: AnalyzeRequest[SsResetCheckRunResult, NoAnalyzeParams],
        *,
        plots: Plots,
    ) -> SsResetCheckAnalyzeResult:
        analysis = ResetCheckExp().analyze(
            req.run_result,
            ResetCheckAnalyzeOptions(
                confusion_matrix=read_confusion_matrix(req.parameters)
            ),
            plots=plots,
        )
        return SsResetCheckAnalyzeResult(
            analysis.reset_mean_excited_population,
            analysis.reset_max_excited_population,
            analysis.reset_max_other_population,
            analysis.worst_sample_gain,
            analysis.analyzed_reset_points,
        )

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_ss_reset_check_{time.strftime('%m%d')}"
