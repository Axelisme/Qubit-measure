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
from zcu_lab.v2.twotone.ro_optimize.length.core import (
    LengthAnalyzeOptions,
    LengthCfg,
    LengthExp,
    LengthResult,
)

RoOptLengthRunResult: TypeAlias = RunRecord[LengthCfg, LengthResult]


@dataclass
class RoOptLengthAnalyzeParams:
    smooth_method: Annotated[
        Literal["wavelet", "gaussian"], ParamMeta(label="Smooth method")
    ] = "wavelet"
    smooth: Annotated[float, ParamMeta(label="Smooth strength", decimals=2)] = 1.0
    # GUI-facing name for LengthAnalyzeOptions.t0: this is a duration
    # normalization overhead, not a penalty strength. Small positive values
    # produce stronger short-readout bias than large positive values.
    duration_t0: Annotated[float | None, ParamMeta(label="Duration t0 (us)")] = None


@dataclass
class RoOptLengthAnalyzeResult(AnalyzeResultBase):
    best_length: float


class RoOptLengthAdapter(
    BaseAdapter[
        LengthCfg,
        RoOptLengthRunResult,
        RoOptLengthAnalyzeResult,
        RoOptLengthAnalyzeParams,
    ]
):
    experiment_tag: ClassVar[str | None] = LengthExp.persistence_tag()
    exp_cls = LengthExp
    ExpCfg_cls: ClassVar[Any] = LengthCfg

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder(
                role_factories=ROLE_FACTORIES,
            )
            .reset(optional=True)
            .pulse("qub_pulse", role_id="pi_pulse")
            .readout(pulse_only=True, locked={"ro_cfg.ro_length": 0.0})
            .relax_delay(
                scaled_path_seed("$qubit.t1", factor=5.0, fallback_value=100.0)
            )
            .sweep(
                "length",
                label="Readout length (us)",
                default=SweepValue(start=0.01, stop=3.5, expts=51),
            )
            .float("skew_penalty", label="Skew penalty", default=0.0, decimals=3)
            .reps(10000)
            .rounds(1)
            .build()
        )

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> RoOptLengthRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        return RunRecord(cfg, LengthExp().run(cfg, context=context))

    def analyze(
        self,
        req: AnalyzeRequest[RoOptLengthRunResult, RoOptLengthAnalyzeParams],
        *,
        plots: Plots,
    ) -> RoOptLengthAnalyzeResult:
        params = req.analyze_params
        options = LengthAnalyzeOptions(
            t0=params.duration_t0,
            smooth=params.smooth,
            smooth_method=params.smooth_method,
        )
        result = LengthExp().analyze(req.run_result, options, plots=plots)
        return RoOptLengthAnalyzeResult(
            best_length=result.best_length,
        )

    def get_writeback_items(
        self, req: WritebackRequest[RoOptLengthRunResult, RoOptLengthAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        items: list[WritebackItem] = [
            ParameterWriteback(
                target_name="best_ro_length",
                description="Optimal readout length (us)",
                proposed_value=result.best_length,
            ),
        ]
        items.extend(
            readout_dpm_writeback_items(
                req.ctx,
                req.run_result.cfg,
                proposed={"best_ro_length": result.best_length},
            )
        )
        return items

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_ro_opt_length_{time.strftime('%m%d')}"
