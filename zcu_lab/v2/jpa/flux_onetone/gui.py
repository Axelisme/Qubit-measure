"""jpa/flux_onetone GUI adapter — JPA flux × readout-frequency 2D survey.

Owns the jpa/flux_onetone cfg definition, run/no-analysis policy and the
operator guide. The core experiment lives in
``zcu_lab.v2.jpa`` (``OneToneFluxExp``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
)

from zcu_lab.v2.jpa.flux_onetone.core import (
    OneToneFluxCfg,
    OneToneFluxExp,
    OneToneFluxResult,
)
from zcu_lab.v2.jpa.flux_onetone.form import make_form

JpaFluxOneToneRunResult: TypeAlias = RunRecord[OneToneFluxCfg, OneToneFluxResult]


def run(
    cfg: OneToneFluxCfg, *, context: RunContext
) -> RunRecord[OneToneFluxCfg, OneToneFluxResult]:
    """Acquire with cfg and retain it as the analysis and save source."""
    return RunRecord(cfg=cfg, result=OneToneFluxExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=OneToneFluxExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
)
