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
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.defaults.role_factories import ROLE_FACTORIES
from zcu_lab.v2._support.measure.seeds import res_freq_range
from zcu_lab.v2._support.measure.writeback_helpers import readout_dpm_writeback_items
from zcu_lab.v2.twotone.ro_optimize.freq.core import (
    FreqAnalyzeOptions,
    FreqCfg,
    FreqExp,
    FreqResult,
)

RoOptFreqRunResult: TypeAlias = RunRecord[FreqCfg, FreqResult]


@dataclass
class RoOptFreqAnalyzeParams:
    smooth_method: Annotated[
        Literal["wavelet", "gaussian"], ParamMeta(label="Smooth method")
    ] = "wavelet"
    smooth: Annotated[float, ParamMeta(label="Smooth strength", decimals=2)] = 2.0


@dataclass
class RoOptFreqAnalyzeResult(AnalyzeResultBase):
    best_freq: float


class RoOptFreqAdapter(
    BaseAdapter[
        FreqCfg,
        RoOptFreqRunResult,
        RoOptFreqAnalyzeResult,
        RoOptFreqAnalyzeParams,
    ]
):
    experiment_tag: ClassVar[str | None] = FreqExp.persistence_tag()
    exp_cls = FreqExp
    ExpCfg_cls: ClassVar[Any] = FreqCfg

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder(
                role_factories=ROLE_FACTORIES,
            )
            .reset(optional=True)
            .pulse("qub_pulse", role_id="pi_pulse")
            .readout(
                pulse_only=True,
                locked={"pulse_cfg.freq": 0.0, "ro_cfg.ro_freq": 0.0},
            )
            .relax_delay(scaled_path_seed("$qubit.t1", factor=5.0, fallback_value=30.5))
            .sweep(
                "freq",
                label="Readout freq (MHz)",
                default=res_freq_range(expts=301, span_factor=1.0),
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
    ) -> RoOptFreqRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        return RunRecord(cfg, FreqExp().run(cfg, context=context))

    def analyze(
        self,
        req: AnalyzeRequest[RoOptFreqRunResult, RoOptFreqAnalyzeParams],
        *,
        plots: Plots,
    ) -> RoOptFreqAnalyzeResult:
        params = req.analyze_params
        options = FreqAnalyzeOptions(
            smooth=params.smooth,
            smooth_method=params.smooth_method,
        )
        result = FreqExp().analyze(req.run_result, options, plots=plots)
        return RoOptFreqAnalyzeResult(
            best_freq=result.best_freq,
        )

    def get_writeback_items(
        self, req: WritebackRequest[RoOptFreqRunResult, RoOptFreqAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        items: list[WritebackItem] = [
            ParameterWriteback(
                target_name="best_ro_freq",
                description="Optimal readout frequency (MHz)",
                proposed_value=result.best_freq,
            ),
        ]
        items.extend(
            readout_dpm_writeback_items(
                req.ctx,
                req.run_result.cfg,
                proposed={"best_ro_freq": result.best_freq},
            )
        )
        return items

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_ro_opt_freq_{time.strftime('%m%d')}"
