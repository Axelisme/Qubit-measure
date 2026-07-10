from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias

from zcu_tools.experiment.v2.singleshot import CheckCfg, CheckExp
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    FigureOnlyAnalyzeResult,
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    ModuleInit,
    scaled_md,
)
from zcu_tools.gui.app.main.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    ExpContext,
    NoAnalyzeParams,
)

from ._shared import read_ge_centers

CheckRunResult: TypeAlias = Any  # CheckResult (frozen domain dataclass)


@dataclass
class CheckAnalyzeResult(FigureOnlyAnalyzeResult):
    # The check is look-at-the-scatter: the domain analyze renders the IQ scatter
    # with the |g>/|e> classification circles and the g/e/other percentages, and
    # extracts no writeback-able scalar. ``figure`` is inherited.
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

    def analyze(
        self, req: AnalyzeRequest[CheckRunResult, NoAnalyzeParams]
    ) -> CheckAnalyzeResult:
        # The check classifies the scatter against the GE centres — read the trio
        # from md (fast-fail if the upstream 'singleshot/ge' writeback is absent).
        g_center, e_center, radius = read_ge_centers(req.md)
        fig = CheckExp().analyze(g_center, e_center, radius, result=req.run_result)
        return CheckAnalyzeResult(figure=fig)

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_sh_check_{time.strftime('%m%d')}"
