from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TypeAlias

from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    AnalyzeResultBase,
    ParameterWriteback,
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
from zcu_lab.v2._support.measure.seeds import res_freq_range
from zcu_lab.v2._support.measure.writeback_helpers import readout_dpm_writeback_items
from zcu_lab.v2.twotone.ro_optimize.auto_optimize.core import (
    AutoOptCfg,
    AutoOptExp,
    AutoOptResult,
)

RoOptAutoRunResult: TypeAlias = RunRecord[AutoOptCfg, AutoOptResult]


@dataclass
class RoOptAutoAnalyzeParams:
    # analyze() takes no tuning knobs — it reads off the best of the optimized
    # points. No form fields.
    pass


@dataclass
class RoOptAutoAnalyzeResult(AnalyzeResultBase):
    best_freq: float
    best_gain: float
    best_length: float


class RoOptAutoAdapter(
    BaseAdapter[
        AutoOptCfg,
        RoOptAutoRunResult,
        RoOptAutoAnalyzeResult,
        RoOptAutoAnalyzeParams,
    ]
):
    experiment_tag: ClassVar[str | None] = AutoOptExp.persistence_tag()
    exp_cls = AutoOptExp
    ExpCfg_cls = AutoOptCfg

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        sweep_label = "Search bounds (min–max)"
        return (
            MeasureCfgBuilder(
                role_factories=ROLE_FACTORIES,
            )
            .reset(optional=True)
            .pulse("qub_pulse", role_id="pi_pulse")
            .readout(
                pulse_only=True,
                locked={
                    "pulse_cfg.freq": 0.0,
                    "ro_cfg.ro_freq": 0.0,
                    "pulse_cfg.gain": 0.0,
                },
            )
            .relax_delay(scaled_path_seed("$qubit.t1", factor=3.0, fallback_value=30.5))
            .sweep(
                "freq",
                label="Readout freq (MHz)",
                default=res_freq_range(expts=51, span_factor=0.2),
                section_label=sweep_label,
            )
            .sweep(
                "gain",
                label="Readout gain (a.u.)",
                default=SweepValue(start=0.1, stop=0.25, expts=51),
                section_label=sweep_label,
            )
            .sweep(
                "length",
                label="Readout length (us)",
                default=SweepValue(start=5.0, stop=10.0, expts=51),
                section_label=sweep_label,
            )
            .int("num_points", label="Optimizer points", default=1001)
            .float("skew_penalty", label="Skew penalty", default=0.0, decimals=3)
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
    ) -> RoOptAutoRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        return RunRecord(cfg, AutoOptExp().run(cfg, context=context))

    def analyze(
        self,
        req: AnalyzeRequest[RoOptAutoRunResult, RoOptAutoAnalyzeParams],
        *,
        plots: Plots,
    ) -> RoOptAutoAnalyzeResult:
        result = AutoOptExp().analyze(req.run_result, None, plots=plots)
        return RoOptAutoAnalyzeResult(
            best_freq=result.best_freq,
            best_gain=result.best_gain,
            best_length=result.best_length,
        )

    def get_writeback_items(
        self, req: WritebackRequest[RoOptAutoRunResult, RoOptAutoAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        items: list[WritebackItem] = [
            ParameterWriteback(
                target_name="best_ro_freq",
                description="Optimal readout frequency (MHz)",
                proposed_value=result.best_freq,
            ),
            ParameterWriteback(
                target_name="best_ro_gain",
                description="Optimal readout gain (a.u.)",
                proposed_value=result.best_gain,
            ),
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
                proposed={
                    "best_ro_freq": result.best_freq,
                    "best_ro_gain": result.best_gain,
                    "best_ro_length": result.best_length,
                },
            )
        )
        return items

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_ro_opt_auto_{time.strftime('%m%d')}"
