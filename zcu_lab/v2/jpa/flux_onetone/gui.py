"""jpa/flux_onetone GUI adapter — JPA flux × readout-frequency 2D survey.

Owns the jpa/flux_onetone cfg definition, run/no-analysis policy and the
operator guide. The core experiment lives in
``zcu_lab.v2.jpa`` (``OneToneFluxExp``).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    RunRequest,
    SessionEnv,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.jpa.flux_onetone.core import (
    OneToneFluxCfg,
    OneToneFluxExp,
    OneToneFluxResult,
)
from zcu_lab.v2.jpa.flux_onetone.form import make_form

JpaFluxOneToneRunResult: TypeAlias = RunRecord[OneToneFluxCfg, OneToneFluxResult]


class JpaFluxOneToneAdapter(BaseAdapter[OneToneFluxCfg, JpaFluxOneToneRunResult]):
    experiment_tag: ClassVar[str | None] = OneToneFluxExp.persistence_tag()
    exp_cls = OneToneFluxExp

    @staticmethod
    def make_form(ctx: FormContext) -> Form[OneToneFluxCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.NONE, load_data=True
    )

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    def run(
        self,
        req: RunRequest,
        cfg: OneToneFluxCfg,
        *,
        context: RunContext,
    ) -> JpaFluxOneToneRunResult:
        del req
        return RunRecord(cfg, OneToneFluxExp().run(cfg, context=context))

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_jpa_flux_onetone_{time.strftime('%m%d')}"
