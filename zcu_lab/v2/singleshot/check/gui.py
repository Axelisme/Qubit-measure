from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_lab.v2.singleshot.check.core import CheckAnalyzeOptions
from zcu_lab.v2.singleshot.check.core import CheckCfg
from zcu_lab.v2.singleshot.check.core import CheckExp
from zcu_lab.v2.singleshot.check.core import CheckResult
from zcu_lab.v2._support.measure.schema_builder import MeasureCfgBuilder
from zcu_lab.v2._support.measure.schema_builder import MeasureCfgDefinition
from zcu_lab.v2._support.measure.schema_builder import ModuleInit
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
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.singleshot_shared import read_ge_centers

CheckRunResult: TypeAlias = RunRecord[CheckCfg, CheckResult]


@dataclass
class CheckAnalyzeResult(AnalyzeResultBase):
    # Classification scatter is published through Plots; no numeric writeback.
    pass


class CheckAdapter(
    BaseAdapter[CheckCfg, CheckRunResult, CheckAnalyzeResult, NoAnalyzeParams]
):
    exp_cls = CheckExp
    ExpCfg_cls: ClassVar[Any] = CheckCfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Single-shot classification check: takes 'shots' single-shot "
            "readouts of the prepared state and draws the IQ scatter with the "
            "|g>/|e> discrimination circles overlaid, reporting the fraction "
            "classified ground / excited / other. Runs on real hardware; the "
            "run itself needs no centres, but the analysis classifies the "
            "scatter against them."
        ),
        expects_md=(
            "REQUIRES the single-shot discrimination calibration in the "
            "MetaDict for ANALYSIS — run 'singleshot/ge' first and apply its "
            "writeback so 'g_center' / 'e_center' / 'ge_radius' are present; "
            "the analyze classifies the scatter against them and fast-fails if "
            "any is missing. Also reads 't1' to set the relax delay; 'q_f' / "
            "'qub_ch' seed the probe drive."
        ),
        expects_ml=(
            "Needs a probe pulse and a readout module. Optionally references a "
            "calibrated reset and an init pulse — both disabled when no library "
            "entry exists."
        ),
        typical_writeback=(
            "No writeback — the check is a visual diagnostic of the existing "
            "single-shot discrimination."
        ),
        recommended=(
            "Run after 'singleshot/ge'. Use a large 'shots' (~5000+) for a "
            "well-sampled scatter; a tight cluster inside the matching circle "
            "indicates good discrimination."
        ),
    )

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder()
            .reset(optional=True)
            .pulse("init_pulse", role_id="pi_pulse", optional=True)
            .pulse(
                "probe_pulse",
                role_id="qub_probe",
                label="Probe Pulse",
                init=ModuleInit.INLINE,
            )
            .readout()
            .relax_delay(scaled_md("t1", factor=5.0, fallback_value=100.0))
            .int("shots", label="Shots", default=5000)
            .reps(1, locked=True)
            .rounds(1, locked=True)
            .build()
        )

    # Standard run path (BaseAdapter.run): CheckExp.run(soc, soccfg, cfg) needs no
    # centres — the trio is an ANALYZE input, read from md in analyze().

    # No get_analyze_params override: NoAnalyzeParams (4th generic arg).

    def run(
        self, req: RunRequest, raw_cfg: dict[str, object], *, context: RunContext
    ) -> CheckRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        result = CheckExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self, req: AnalyzeRequest[CheckRunResult, NoAnalyzeParams], *, plots: Plots
    ) -> CheckAnalyzeResult:
        # The check classifies the scatter against the GE centres — read the trio
        # from md (fast-fail if the upstream 'singleshot/ge' writeback is absent).
        g_center, e_center, radius = read_ge_centers(req.md)
        CheckExp().analyze(
            req.run_result, CheckAnalyzeOptions(g_center, e_center, radius), plots=plots
        )
        return CheckAnalyzeResult()

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_sh_check_{time.strftime('%m%d')}"
