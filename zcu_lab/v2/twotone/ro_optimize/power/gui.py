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
    SessionEnv,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.app.measure.schema_builder import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
)
from zcu_tools.gui.app.measure.seeds import scaled_path_seed
from zcu_tools.gui.cfg import (
    SweepValue,
)
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.defaults.role_factories import ROLE_FACTORIES
from zcu_lab.v2._support.measure.writeback_helpers import readout_dpm_writeback_items
from zcu_lab.v2.twotone.ro_optimize.power.core import (
    PowerAnalyzeOptions,
    PowerCfg,
    PowerExp,
    PowerResult,
)

RoOptPowerRunResult: TypeAlias = RunRecord[PowerCfg, PowerResult]


@dataclass
class RoOptPowerAnalyzeParams:
    smooth_method: Annotated[
        Literal["wavelet", "gaussian"], ParamMeta(label="Smooth method")
    ] = "wavelet"
    smooth: Annotated[float, ParamMeta(label="Smooth strength", decimals=2)] = 1.0
    penalty_ratio: Annotated[
        float, ParamMeta(label="Power penalty ratio", decimals=2)
    ] = 0.5


@dataclass
class RoOptPowerAnalyzeResult(AnalyzeResultBase):
    best_gain: float


class RoOptPowerAdapter(
    BaseAdapter[
        PowerCfg,
        RoOptPowerRunResult,
        RoOptPowerAnalyzeResult,
        RoOptPowerAnalyzeParams,
    ]
):
    experiment_tag: ClassVar[str | None] = PowerExp.persistence_tag()
    exp_cls = PowerExp
    ExpCfg_cls: ClassVar[Any] = PowerCfg

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder(
                role_factories=ROLE_FACTORIES,
            )
            .reset(optional=True)
            .pulse("qub_pulse", role_id="pi_pulse")
            .readout(pulse_only=True, locked={"pulse_cfg.gain": 0.0})
            .relax_delay(
                scaled_path_seed("$qubit.t1", factor=5.0, fallback_value=100.0)
            )
            .sweep(
                "gain",
                label="Readout gain (a.u.)",
                default=SweepValue(start=0.001, stop=0.2, expts=101),
            )
            .float("skew_penalty", label="Skew penalty", default=0.0, decimals=3)
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
    ) -> RoOptPowerRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        return RunRecord(cfg, PowerExp().run(cfg, context=context))

    def analyze(
        self,
        req: AnalyzeRequest[RoOptPowerRunResult, RoOptPowerAnalyzeParams],
        *,
        plots: Plots,
    ) -> RoOptPowerAnalyzeResult:
        params = req.analyze_params
        options = PowerAnalyzeOptions(
            penalty_ratio=params.penalty_ratio,
            smooth=params.smooth,
            smooth_method=params.smooth_method,
        )
        result = PowerExp().analyze(req.run_result, options, plots=plots)
        return RoOptPowerAnalyzeResult(
            best_gain=result.best_gain,
        )

    def get_writeback_items(
        self, req: WritebackRequest[RoOptPowerRunResult, RoOptPowerAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        items: list[WritebackItem] = [
            ParameterWriteback(
                target_name="best_ro_gain",
                description="Optimal readout gain (a.u.)",
                proposed_value=result.best_gain,
            ),
        ]
        items.extend(
            readout_dpm_writeback_items(
                req.ctx,
                req.run_result.cfg,
                proposed={"best_ro_gain": result.best_gain},
            )
        )
        return items

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_ro_opt_gain_{time.strftime('%m%d')}"
