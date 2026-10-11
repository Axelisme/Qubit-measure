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
    exp_cls = LengthExp
    ExpCfg_cls: ClassVar[Any] = LengthCfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Bath-reset length sweep: holds the tested bath reset at its "
            "calibrated cavity frequency and gain and sweeps its duration "
            "(cavity and qubit tones together), with an internal 4-point pi/2 "
            "tomography phase axis, showing how the residual excitation decays "
            "with reset length. Runs on real hardware. Run after the cavity "
            "freq/gain are found, to pick the shortest fully-resetting length."
        ),
        expects_md=(
            "Requires $qubit.ext.reset_bath_freq and "
            "$qubit.ext.reset_bath_gain for the tested cavity tone. Missing, "
            "null or invalid values fail cfg lowering. Other drive defaults "
            "use captured qubit/resonator frequency and wiring."
        ),
        expects_ml=(
            "Builds the tested bath-reset cfg from captured defaults and "
            "needs a readout module. Optionally references an upstream reset and an "
            "init pulse (a library pi pulse when present) — both disabled when "
            "no library entry exists."
        ),
        typical_writeback=(
            "No writeback — the chosen reset length is read off the decay "
            "curve by eye, and registering the calibrated 'reset_bath' module "
            "is left to the final phase step."
        ),
        recommended=(
            "A length sweep from ~0.05 us to a few times the expected reset "
            "time captures the full decay; shorten the span once the plateau "
            "is clear. Allow a long relax delay so the qubit fully relaxes "
            "before each reset."
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
