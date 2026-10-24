from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
)

from zcu_lab.v2.onetone.power_dep.core import PowerDepCfg, PowerDepExp, PowerDepResult
from zcu_lab.v2.onetone.power_dep.form import make_form

OneTonePowerDepRunResult: TypeAlias = RunRecord[PowerDepCfg, PowerDepResult]


def run(
    cfg: PowerDepCfg, *, context: RunContext
) -> RunRecord[PowerDepCfg, PowerDepResult]:
    """Acquire with cfg and retain it as the analysis and save source."""
    return RunRecord(cfg=cfg, result=PowerDepExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=PowerDepExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
)
