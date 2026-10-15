from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    AnalyzeResultBase,
    NoAnalyzeParams,
    RunRequest,
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

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

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
