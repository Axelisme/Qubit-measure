from __future__ import annotations

from pathlib import Path
from typing import ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    RunRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.twotone.power_dep.core import PowerCfg, PowerExp, PowerResult

from .form import make_form

PowerDepRunResult: TypeAlias = RunRecord[PowerCfg, PowerResult]


class PowerDepAdapter(BaseAdapter[PowerCfg, PowerDepRunResult]):
    @staticmethod
    def make_form(ctx: FormContext) -> Form[PowerCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    experiment_tag: ClassVar[str | None] = PowerExp.persistence_tag()
    exp_cls = PowerExp
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.NONE, load_data=True
    )

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    def run(
        self,
        req: RunRequest,
        cfg: PowerCfg,
        *,
        context: RunContext,
    ) -> PowerDepRunResult:
        del req
        result = PowerExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)
