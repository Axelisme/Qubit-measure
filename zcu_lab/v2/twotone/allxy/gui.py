"""AllXY gate-error check adapter.

Runs the 21 standard AllXY gate pairs and reuses ``AllXY_Exp.analyze``, which
fits power and detuning errors; the summary reports them.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, ClassVar, TypeAlias

from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    AnalyzeResultBase,
    ParamMeta,
    RunRequest,
    SessionEnv,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.app.measure.schema_builder import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
)
from zcu_tools.gui.app.measure.seeds import scaled_path_seed
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.defaults.role_factories import ROLE_FACTORIES
from zcu_lab.v2.twotone.allxy.core import (
    AllXY_Exp,
    AllXY_Result,
    AllXYAnalyzeOptions,
    AllXYCfg,
)

AllXYRunResult: TypeAlias = RunRecord[AllXYCfg, AllXY_Result]


@dataclass
class AllXYAnalyzeParams:
    fit_ge: Annotated[bool, ParamMeta(label="Fit g/e levels")] = False


@dataclass
class AllXYAnalyzeResult(AnalyzeResultBase):
    amplitude_error: float
    detune_param: float
    power_err: float
    detune_err: float
    residual_rms: float


class AllXYAdapter(
    BaseAdapter[AllXYCfg, AllXYRunResult, AllXYAnalyzeResult, AllXYAnalyzeParams]
):
    experiment_tag: ClassVar[str | None] = AllXY_Exp.persistence_tag()
    exp_cls = AllXY_Exp
    ExpCfg_cls: ClassVar[Any] = AllXYCfg

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder(
                role_factories=ROLE_FACTORIES,
            )
            .reset(optional=True)
            .pulse("X90_pulse", role_id="pi2_pulse", label="X90 Pulse")
            .pulse("X180_pulse", role_id="pi_pulse", label="X180 Pulse")
            .readout()
            .relax_delay(scaled_path_seed("$qubit.t1", factor=5.0, fallback_value=30.5))
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
    ) -> AllXYRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        return RunRecord(cfg=cfg, result=AllXY_Exp().run(cfg, context=context))

    def analyze(
        self,
        req: AnalyzeRequest[AllXYRunResult, AllXYAnalyzeParams],
        *,
        plots: Plots,
    ) -> AllXYAnalyzeResult:
        analysis = AllXY_Exp().analyze(
            req.run_result,
            AllXYAnalyzeOptions(fit_ge=req.analyze_params.fit_ge),
            plots=plots,
        )
        return AllXYAnalyzeResult(
            amplitude_error=analysis.amplitude_error,
            detune_param=analysis.detune_param,
            power_err=analysis.power_err,
            detune_err=analysis.detune_err,
            residual_rms=analysis.residual_rms,
        )

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_allxy_{time.strftime('%m%d')}"
