"""Measure GUI wiring for raw two-generator STIRAP scans."""

from pathlib import Path

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter

from .core import StirapCfg, StirapExp, StirapResult
from .form import make_form


def _run(cfg: StirapCfg, *, context: RunContext) -> RunRecord[StirapCfg, StirapResult]:
    return RunRecord(cfg=cfg, result=StirapExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=StirapExp.AXES_SPEC,
    make_form=make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
)
