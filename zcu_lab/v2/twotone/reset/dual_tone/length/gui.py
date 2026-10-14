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
from zcu_tools.gui.app.measure.seeds import path_seed
from zcu_tools.gui.cfg import (
    SweepValue,
)
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.defaults.role_factories import ROLE_FACTORIES
from zcu_lab.v2.twotone.reset.dual_tone.length.core import (
    LengthCfg,
    LengthExp,
    LengthResult,
)

DualToneLengthRunResult: TypeAlias = RunRecord[LengthCfg, LengthResult]


@dataclass
class DualToneLengthAnalyzeResult(AnalyzeResultBase):
    """Length analysis publishes a curve without fitted scalar output."""

    pass


class DualToneLengthAdapter(
    BaseAdapter[
        LengthCfg,
        DualToneLengthRunResult,
        DualToneLengthAnalyzeResult,
        NoAnalyzeParams,
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
            .pulse("init_pulse", role_id="pi_pulse", optional=True)
            .reset(
                "tested_reset",
                role_id="two_pulse_reset",
                label="Tested Reset",
                shape="two_pulse",
                overrides={
                    "pulse1_cfg.freq": path_seed("$qubit.ext.reset_f1", fallback=0.0),
                    "pulse2_cfg.freq": path_seed("$qubit.ext.reset_f2", fallback=0.0),
                    "pulse1_cfg.gain": path_seed(
                        "$qubit.ext.reset_gain1", fallback=1.0
                    ),
                    "pulse2_cfg.gain": path_seed(
                        "$qubit.ext.reset_gain2", fallback=1.0
                    ),
                },
            )
            .readout()
            .relax_delay(0.5)
            .sweep(
                "length",
                label="Length (us)",
                default=SweepValue(start=0.05, stop=40.0, expts=51),
            )
            .reps(100)
            .rounds(100)
            .build()
        )

    # No get_analyze_params override: NoAnalyzeParams (the 4th generic arg) makes
    # BaseAdapter return the empty params instance and reflect the type.

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> DualToneLengthRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        result = LengthExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self,
        req: AnalyzeRequest[DualToneLengthRunResult, NoAnalyzeParams],
        *,
        plots: Plots,
    ) -> DualToneLengthAnalyzeResult:
        LengthExp().analyze(req.run_result, None, plots=plots)
        return DualToneLengthAnalyzeResult()

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_dualreset_length_{time.strftime('%m%d')}"
