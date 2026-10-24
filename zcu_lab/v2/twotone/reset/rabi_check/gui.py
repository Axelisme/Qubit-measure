from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis, AnalyzeRequest
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.twotone.reset.rabi_check.core import (
    RabiCheckCfg,
    RabiCheckExp,
    RabiCheckResult,
)
from zcu_lab.v2.twotone.reset.rabi_check.fit import RabiCheckFit
from zcu_lab.v2.twotone.reset.rabi_check.form import make_form

RabiCheckRunResult: TypeAlias = RunRecord[RabiCheckCfg, RabiCheckResult]


def _RabiCheck_analyze(
    req: AnalyzeRequest[RabiCheckRunResult, None], *, plots: Plots
) -> RabiCheckFit:
    return RabiCheckExp().analyze(req.run_result, None, plots=plots)


def _RabiCheck_make_form(ctx: FormContext) -> Form[RabiCheckCfg]:
    """Build this adapter's definition using captured Open/Reset sources."""
    return make_form(ctx)


def _run(
    cfg: RabiCheckCfg, *, context: RunContext
) -> RunRecord[RabiCheckCfg, RabiCheckResult]:
    """Acquire a trace and retain the experiment configuration with its result."""
    return RunRecord(cfg=cfg, result=RabiCheckExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=RabiCheckExp.AXES_SPEC,
    make_form=_RabiCheck_make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_RabiCheck_analyze),
)
