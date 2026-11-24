"""Measure attachment for heralded idle ZZ; acquisition never writes calibration."""

from pathlib import Path
from typing import TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter

from .core import IdleZZHeraldedCfg, IdleZZHeraldedExp, IdleZZHeraldedResult
from .form import make_form

IdleZZHeraldedRun: TypeAlias = RunRecord[IdleZZHeraldedCfg, IdleZZHeraldedResult]


def run(
    cfg: IdleZZHeraldedCfg, *, context: RunContext
) -> RunRecord[IdleZZHeraldedCfg, IdleZZHeraldedResult]:
    """Run the experiment and retain its actual configuration."""
    return RunRecord(cfg=cfg, result=IdleZZHeraldedExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=IdleZZHeraldedExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
)
