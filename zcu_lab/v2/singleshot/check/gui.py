from __future__ import annotations

import time
from dataclasses import dataclass
from typing import ClassVar, TypeAlias

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
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.singleshot_shared import read_ge_centers
from zcu_lab.v2.singleshot.check.core import (
    CheckAnalyzeOptions,
    CheckCfg,
    CheckExp,
    CheckResult,
)
from zcu_lab.v2.singleshot.check.form import make_form

CheckRunResult: TypeAlias = RunRecord[CheckCfg, CheckResult]


@dataclass
class CheckAnalyzeResult(AnalyzeResultBase):
    # Classification scatter is published through Plots; no numeric writeback.
    pass


class CheckAdapter(
    BaseAdapter[CheckCfg, CheckRunResult, CheckAnalyzeResult, NoAnalyzeParams]
):
    experiment_tag: ClassVar[str | None] = CheckExp.persistence_tag()

    @staticmethod
    def make_form(ctx: FormContext) -> Form[CheckCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    exp_cls = CheckExp

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Single-shot classification check: takes 'shots' single-shot readouts of the "
            "prepared state and draws the IQ scatter with the |g>/|e> discrimination "
            "circles overlaid, reporting the fraction classified ground / excited / "
            "other. Runs on real hardware; the run itself needs no centres, but the "
            "analysis classifies the scatter against them."
        ),
        expects_md=(
            "Open and Reset capture the current aliases, parameters and library. Relax "
            "delay is 5 * $qubit.t1. Fresh inline probe uses $qubit.freq and "
            "$qubit.wiring.L01.ch. Inline templates read $qubit.freq, "
            "$qubit.wiring.L01.ch and $resonator.freq/wiring.ch/wiring.ro_ch; readout "
            "trigger uses $resonator.ext.time_of_fly + 0.05 us. Missing required sources "
            "remain Invalid; no numeric fallback is supplied."
        ),
        expects_ml=(
            "Optional init/reset use $qubit.module.x180/reset. Probe starts from "
            "qub_probe. Readout uses $qubit.module.readout or the readout template. Run "
            "uses this fixed capture."
        ),
        typical_writeback=(
            "No writeback — the check is a visual diagnostic of the existing single-shot "
            "discrimination."
        ),
        recommended=(
            "Run after 'singleshot/ge'. Use a large 'shots' (~5000+) for a well-sampled "
            "scatter; a tight cluster inside the matching circle indicates good "
            "discrimination."
        ),
    )

    # Standard run path (BaseAdapter.run): CheckExp.run(soc, soccfg, cfg) needs no
    # centres — the trio is an ANALYZE input, read from md in analyze().

    # No get_analyze_params override: NoAnalyzeParams (4th generic arg).

    def run(
        self,
        req: RunRequest,
        cfg: CheckCfg,
        *,
        context: RunContext,
    ) -> CheckRunResult:
        del req
        result = CheckExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self, req: AnalyzeRequest[CheckRunResult, NoAnalyzeParams], *, plots: Plots
    ) -> CheckAnalyzeResult:
        # The check classifies the scatter against the GE centres — read the trio
        # from md (fast-fail if the upstream 'singleshot/ge' writeback is absent).
        g_center, e_center, radius = read_ge_centers(req.parameters)
        CheckExp().analyze(
            req.run_result, CheckAnalyzeOptions(g_center, e_center, radius), plots=plots
        )
        return CheckAnalyzeResult()

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_sh_check_{time.strftime('%m%d')}"
