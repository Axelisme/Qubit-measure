"""Measure attachment for common-offset mux spectroscopy."""

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

from zcu_lab.v2.mux.onetone.core import (
    MuxOnetoneCfg,
    MuxOnetoneExp,
    MuxOnetoneResult,
)
from zcu_lab.v2.mux.onetone.form import make_form

MuxOnetoneRunResult: TypeAlias = RunRecord[MuxOnetoneCfg, MuxOnetoneResult]


class MuxOnetoneAdapter(BaseAdapter[MuxOnetoneCfg, MuxOnetoneRunResult]):
    """Run and persist active spectra with the complete readout and selection."""

    experiment_tag: ClassVar[str | None] = MuxOnetoneExp.persistence_tag()
    exp_cls = MuxOnetoneExp
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.NONE, load_data=True
    )
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @staticmethod
    def make_form(ctx: FormContext) -> Form[MuxOnetoneCfg]:
        """Build the mux draft from the fixed Open/Reset capture."""
        return make_form(ctx)

    def run(
        self,
        req: RunRequest,
        cfg: MuxOnetoneCfg,
        *,
        context: RunContext,
    ) -> MuxOnetoneRunResult:
        """Acquire with accepted cfg and explicit hardware/plot/cancel context.

        req is unused. Return the original cfg paired with core data, including
        partial acquisition data. Acquisition failures follow the RunContext
        cancel signal contract; construction and configuration errors propagate.
        No current form, alias or library source is read here.
        """
        del req
        result = MuxOnetoneExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)
