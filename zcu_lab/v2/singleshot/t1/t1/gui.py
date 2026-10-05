from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_lab.v2.singleshot.t1.t1.core import T1AnalyzeOptions
from zcu_lab.v2.singleshot.t1.t1.core import T1Cfg
from zcu_lab.v2.singleshot.t1.t1.core import T1Exp
from zcu_lab.v2.singleshot.t1.t1.core import T1Result
from zcu_lab.v2._support.measure.schema_builder import MeasureCfgBuilder
from zcu_lab.v2._support.measure.schema_builder import MeasureCfgDefinition
from zcu_lab.v2._support.measure.seeds import SweepDefault
from zcu_lab.v2._support.measure.seeds import scaled_md
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
    EvalValue,
    ScalarSpec,
)
from zcu_tools.plotting.plots import Plots

# Transition-rate analysis publishes figures without numeric writeback.
SsT1RunResult: TypeAlias = RunRecord[T1Cfg, T1Result]


@dataclass
class SsT1AnalyzeResult(AnalyzeResultBase):
    pass


class SsT1Adapter(
    BaseAdapter[T1Cfg, SsT1RunResult, SsT1AnalyzeResult, NoAnalyzeParams]
):
    exp_cls = T1Exp
    ExpCfg_cls: ClassVar[Any] = T1Cfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Single-shot T1: applies a π pulse, waits for a variable delay, "
            "reads out, and classifies each shot in-program against the |g>/|e> "
            "IQ-cluster centres. Repeats from both |g> and |e> initial states "
            "(Branch). Plots the ground / excited / other populations versus "
            "delay time with dual-transition-rate fits for T1 and T1_b. "
            "Runs on real hardware."
        ),
        expects_md=(
            "Run freezes 'g_center' / 'e_center' / 'ge_radius' from resolved "
            "cfg, not live MetaDict. Enter direct cfg values or optionally seed "
            "defaults with 'singleshot/ge' writeback. Missing or invalid cfg "
            "calibration fails before hardware. "
            "Optionally reads 'confusion_matrix' to readout-correct populations "
            "at analyze time; 't1' to seed the sweep stop (5*t1; "
            "fallback 500 us) and relax delay (5*t1; fallback 100 us); "
            "'q_f' / 'qub_ch' for the pi pulse; 'r_f' / "
            "'res_ch' / 'ro_ch' / 'timeFly' for readout."
        ),
        expects_ml=(
            "Needs a qubit pi-pulse module and a readout module. "
            "Optional reset (disabled when no library entry exists)."
        ),
        typical_writeback=(
            "No writeback — T1 is shown in the plot title only. "
            "Use 'singleshot/t1_tone' if you need the T1 scalar in the "
            "MetaDict."
        ),
        recommended=(
            "Set calibration cfg directly or seed it with 'singleshot/ge'. "
            "A delay sweep reaching ~5*T1 lets the "
            "decay flatten; with no prior 't1', the sweep spans 0–500 us. "
            "Set 'uniform=True' to sweep linearly; leave False to cluster more "
            "points along the expected exponential decay while preserving the "
            "configured start/stop window and point count."
        ),
    )

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder()
            .reset(optional=True)
            .pulse("pi_pulse", role_id="pi_pulse")
            .readout()
            .relax_delay(scaled_md("t1", factor=5.0, fallback_value=100.0))
            .sweep(
                "length",
                label="Delay (us)",
                default=SweepDefault(
                    start=0.0,
                    # The fallback is the fully scaled 5 * 100 us window.
                    stop=scaled_md("t1", factor=5.0, fallback_value=500.0),
                    expts=101,
                ),
            )
            .field(
                "g_center",
                spec=ScalarSpec("Ground center", complex),
                default=EvalValue("g_center"),
            )
            .field(
                "e_center",
                spec=ScalarSpec("Excited center", complex),
                default=EvalValue("e_center"),
            )
            .field(
                "radius",
                spec=ScalarSpec("Classification radius", float),
                default=EvalValue("ge_radius"),
            )
            .bool("uniform", label="Uniform (linear) sweep", default=False)
            .reps(1000)
            .rounds(10)
            .build()
        )

    def run(
        self, req: RunRequest, raw_cfg: dict[str, object], *, context: RunContext
    ) -> SsT1RunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        return RunRecord(cfg, T1Exp().run(cfg, context=context))

    def analyze(
        self, req: AnalyzeRequest[SsT1RunResult, NoAnalyzeParams], *, plots: Plots
    ) -> SsT1AnalyzeResult:
        # ``confusion_matrix`` is the GE 3×3 readout-correction matrix from md.
        # ``skip`` is not exposed as a user knob — users can re-run with a shorter
        # sweep instead.
        confusion = req.md.get("confusion_matrix")
        T1Exp().analyze(
            req.run_result, T1AnalyzeOptions(confusion_matrix=confusion), plots=plots
        )
        return SsT1AnalyzeResult()

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_ss_t1_{time.strftime('%m%d')}"
