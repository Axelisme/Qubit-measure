from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
)
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.twotone.reset.bath.length.core import LengthCfg, LengthExp, LengthResult
from zcu_lab.v2.twotone.reset.bath.length.form import make_form

BathLengthRunResult: TypeAlias = RunRecord[LengthCfg, LengthResult]


def _BathLength_make_form(ctx: FormContext) -> Form[LengthCfg]:
    """Build this adapter's definition using captured Open/Reset sources."""
    return make_form(ctx)


def _BathLength_analyze(
    req: AnalyzeRequest[BathLengthRunResult, None],
    *,
    plots: Plots,
) -> None:
    LengthExp().analyze(req.run_result, None, plots=plots)
    return


DEFINITION = Adapter(
    spec=LengthExp.AXES_SPEC,
    make_form=_BathLength_make_form,
    run=LengthExp().run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_BathLength_analyze),
)
