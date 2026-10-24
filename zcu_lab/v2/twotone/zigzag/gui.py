"""GUI adapter for the plain ZigZag repetition experiment."""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
)
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.twotone.zigzag.core import ZigZagCfg, ZigZagExp, ZigZagResult

from .form import make_form

ZigZagRunResult: TypeAlias = RunRecord[ZigZagCfg, ZigZagResult]


def _ZigZag_make_form(ctx: FormContext) -> Form[ZigZagCfg]:
    """Build this adapter's definition using captured Open/Reset sources."""
    return make_form(ctx)


DEFINITION = Adapter(
    spec=ZigZagExp.AXES_SPEC,
    make_form=_ZigZag_make_form,
    run=ZigZagExp().run,
    guide=Path(__file__).with_name("guide.md"),
)
