"""jpa/check GUI adapter — JPA pump off/on resonator comparison check.

Owns the jpa/check cfg definition, run/analyze/writeback policy and the
operator guide. The core experiment lives in ``zcu_lab.v2.jpa``
(``CheckExp``).
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    RunRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.jpa.check.core import CheckCfg, CheckExp, CheckResult
from zcu_lab.v2.jpa.check.form import make_form

JpaCheckRunResult: TypeAlias = RunRecord[CheckCfg, CheckResult]


class JpaCheckAdapter(BaseAdapter[CheckCfg, JpaCheckRunResult]):
    exp_cls = CheckExp

    @staticmethod
    def make_form(ctx: FormContext) -> Form[CheckCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    def run(
        self,
        req: RunRequest,
        cfg: CheckCfg,
        *,
        context: RunContext,
    ) -> JpaCheckRunResult:
        del req
        return RunRecord(cfg, CheckExp().run(cfg, context=context))

    def analyze(
        self, req: AnalyzeRequest[JpaCheckRunResult, None], *, plots: Plots
    ) -> None:
        CheckExp().analyze(req.run_result, None, plots=plots)
        return
