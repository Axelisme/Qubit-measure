"""GUI adapter for the plain ZigZag repetition experiment."""

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

from zcu_lab.v2.twotone.zigzag.core import ZigZagCfg, ZigZagExp, ZigZagResult

from .form import make_form

ZigZagRunResult: TypeAlias = RunRecord[ZigZagCfg, ZigZagResult]


class ZigZagAdapter(BaseAdapter[ZigZagCfg, ZigZagRunResult]):
    @staticmethod
    def make_form(ctx: FormContext) -> Form[ZigZagCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    experiment_tag: ClassVar[str | None] = ZigZagExp.persistence_tag()
    exp_cls = ZigZagExp
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.NONE, load_data=True
    )

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    def run(
        self,
        req: RunRequest,
        cfg: ZigZagCfg,
        *,
        context: RunContext,
    ) -> ZigZagRunResult:
        del req
        result = ZigZagExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_zigzag_{time.strftime('%m%d')}"
