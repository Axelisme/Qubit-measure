from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
)
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.twotone.power_dep.core import PowerCfg, PowerExp, PowerResult

from .form import make_form

PowerDepRunResult: TypeAlias = RunRecord[PowerCfg, PowerResult]


def _PowerDep_make_form(ctx: FormContext) -> Form[PowerCfg]:
    """Build this adapter's definition using captured Open/Reset sources."""
    return make_form(ctx)


DEFINITION = Adapter(
    spec=PowerExp.AXES_SPEC,
    make_form=_PowerDep_make_form,
    run=PowerExp().run,
    guide=Path(__file__).with_name("guide.md"),
)
