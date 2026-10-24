"""jpa/check GUI adapter — JPA pump off/on resonator comparison check.

Owns the jpa/check cfg definition, run/analyze/writeback policy and the
operator guide. The core experiment lives in ``zcu_lab.v2.jpa``
(``CheckExp``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
)
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.jpa.check.core import CheckCfg, CheckExp, CheckResult
from zcu_lab.v2.jpa.check.form import make_form

JpaCheckRunResult: TypeAlias = RunRecord[CheckCfg, CheckResult]


def _analyze(req: AnalyzeRequest[JpaCheckRunResult, None], *, plots: Plots) -> None:
    CheckExp().analyze(req.run_result, None, plots=plots)
    return


def run(cfg: CheckCfg, *, context: RunContext) -> RunRecord[CheckCfg, CheckResult]:
    """Acquire with cfg and retain it as the analysis and save source."""
    return RunRecord(cfg=cfg, result=CheckExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=CheckExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze),
)
