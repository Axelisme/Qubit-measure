from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias

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
    ModuleInit,
)
from zcu_lab.v2._support.measure.seeds import path_seed
from zcu_lab.v2.twotone.reset.single_tone.length.core import (
    LengthCfg,
    LengthExp,
    LengthResult,
)

SingleToneLengthRunResult: TypeAlias = RunRecord[LengthCfg, LengthResult]


@dataclass
class SingleToneLengthAnalyzeResult(AnalyzeResultBase):
    """Length analysis publishes a curve without fitted scalar output."""

    pass


class SingleToneLengthAdapter(
    BaseAdapter[
        LengthCfg,
        SingleToneLengthRunResult,
        SingleToneLengthAnalyzeResult,
        NoAnalyzeParams,
    ]
):
    exp_cls = LengthExp
    ExpCfg_cls: ClassVar[Any] = LengthCfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Single-tone sideband-reset length sweep: holds the tested "
            "pulse-reset at its calibrated sideband frequency and sweeps its "
            "duration, showing how the residual excitation decays with reset "
            "length. Runs on real hardware. Run after the reset frequency is "
            "found, to pick the shortest length that fully resets the qubit."
        ),
        expects_md=(
            "Requires $qubit.ext.reset_f for the tested reset frequency. "
            "Missing, null or invalid values fail cfg lowering. Other drive "
            "defaults use captured qubit frequency and wiring."
        ),
        expects_ml=(
            "Builds the tested pulse reset inline and needs a readout "
            "module. Optionally references a calibrated upstream reset and an "
            "init pulse (a library pi pulse when present) — both disabled when "
            "no library entry exists."
        ),
        typical_writeback=(
            "No automatic writeback. Analysis plots the length response "
            "without fitting a length. Choose the length manually; "
            "SingleToneFreq proposes the tested reset module."
        ),
        recommended=(
            "A length sweep from ~0.1 us to a few times the expected reset "
            "time captures the full decay; shorten the span once the plateau "
            "is clear. Allow a long relax delay between shots so the qubit "
            "fully relaxes before each reset."
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
                role_id="reset",
                label="Tested Reset",
                shape="pulse",
                init=ModuleInit.INLINE,
                overrides={
                    "pulse_cfg.freq": path_seed("$qubit.ext.reset_f", fallback=0.0)
                },
            )
            .readout()
            .relax_delay(30.5)
            .sweep(
                "length",
                label="Length (us)",
                default=SweepValue(start=0.1, stop=20.0, expts=50),
            )
            .reps(100)
            .rounds(100)
            .build()
        )

    # No get_analyze_params override: NoAnalyzeParams (the 4th generic arg) makes
    # BaseAdapter return the empty params instance and reflect the type.

    def run(
        self, req: RunRequest, raw_cfg: dict[str, object], *, context: RunContext
    ) -> SingleToneLengthRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        result = LengthExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self,
        req: AnalyzeRequest[SingleToneLengthRunResult, NoAnalyzeParams],
        *,
        plots: Plots,
    ) -> SingleToneLengthAnalyzeResult:
        LengthExp().analyze(req.run_result, None, plots=plots)
        return SingleToneLengthAnalyzeResult()

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_sidereset_length_{time.strftime('%m%d')}"
