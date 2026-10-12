from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias

from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    NoAnalyzeParams,
    RunRequest,
    SessionEnv,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.cfg import (
    SweepValue,
)
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.schema_builder import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
)
from zcu_lab.v2._support.measure.seeds import path_seed
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

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Dual-tone reset length sweep: holds the tested two-pulse reset at "
            "its calibrated sideband frequencies and gains and sweeps its "
            "duration (both tones together), showing how the residual "
            "excitation decays with reset length. Runs on real hardware. Run "
            "last, to pick the shortest length that fully resets the qubit."
        ),
        expects_md=(
            "Requires $qubit.ext.reset_f1/reset_f2 and "
            "$qubit.ext.reset_gain1/reset_gain2 for the tested frequencies "
            "and gains. Missing, null or invalid values fail cfg lowering. "
            "Other drive defaults use captured qubit frequency and wiring."
        ),
        expects_ml=(
            "Builds the tested two-pulse-reset cfg from captured defaults and "
            "needs a readout module. Optionally references an upstream reset and an "
            "init pulse (a library pi pulse when present) — both disabled when "
            "no library entry exists."
        ),
        typical_writeback=(
            "No automatic writeback. Analysis plots the length response "
            "without fitting a length. Choose the length manually; "
            "DualTonePower proposes the tested reset module."
        ),
        recommended=(
            "A length sweep from ~0.05 us to a few times the expected reset "
            "time captures the full decay; shorten the span once the plateau "
            "is clear."
        ),
    )

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder()
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
