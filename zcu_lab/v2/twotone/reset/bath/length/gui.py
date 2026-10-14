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
from zcu_lab.v2.twotone.reset.bath.length.core import LengthCfg, LengthExp, LengthResult

BathLengthRunResult: TypeAlias = RunRecord[LengthCfg, LengthResult]

# The domain replaces pi2_cfg.phase with a 4-point QickSweep1D("phase", 0, 270)
# for the tomography readout, so the form value is inert; lock it to the
# notebook offset to keep the form from showing a field the run overwrites.
_PI2_PHASE_OFFSET_DEG: float = 90.0


@dataclass
class BathLengthAnalyzeResult(AnalyzeResultBase):
    """The length analysis publishes a figure without fitted scalars."""


class BathLengthAdapter(
    BaseAdapter[
        LengthCfg,
        BathLengthRunResult,
        BathLengthAnalyzeResult,
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
                role_id="bath_reset",
                label="Tested Reset",
                shape="bath",
                locked={"pi2_cfg.phase": _PI2_PHASE_OFFSET_DEG},
                overrides={
                    "cavity_tone_cfg.freq": path_seed(
                        "$qubit.ext.reset_bath_freq", fallback=0.0
                    ),
                    "cavity_tone_cfg.gain": path_seed(
                        "$qubit.ext.reset_bath_gain", fallback=0.1
                    ),
                },
            )
            .readout()
            .relax_delay(10.5)
            .sweep(
                "length",
                label="Length (us)",
                default=SweepValue(start=0.05, stop=15.0, expts=201),
            )
            .reps(100)
            .rounds(1000)
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
    ) -> BathLengthRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        result = LengthExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self,
        req: AnalyzeRequest[BathLengthRunResult, NoAnalyzeParams],
        *,
        plots: Plots,
    ) -> BathLengthAnalyzeResult:
        LengthExp().analyze(req.run_result, None, plots=plots)
        return BathLengthAnalyzeResult()

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_bathreset_len_{time.strftime('%m%d')}"
