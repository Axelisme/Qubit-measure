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

from zcu_lab.v2.onetone.power_dep.core import PowerDepCfg, PowerDepExp, PowerDepResult
from zcu_lab.v2.onetone.power_dep.form import make_form

OneTonePowerDepRunResult: TypeAlias = RunRecord[PowerDepCfg, PowerDepResult]


class OneTonePowerDepAdapter(BaseAdapter[PowerDepCfg, OneTonePowerDepRunResult]):
    experiment_tag: ClassVar[str | None] = PowerDepExp.persistence_tag()
    exp_cls = PowerDepExp

    @staticmethod
    def make_form(ctx: FormContext) -> Form[PowerDepCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.NONE, load_data=True
    )

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    def run(
        self,
        req: RunRequest,
        cfg: PowerDepCfg,
        *,
        context: RunContext,
    ) -> OneTonePowerDepRunResult:
        """Acquire a power/frequency map from accepted cfg values.

        cfg is this adapter's validated Form output. earlystop_snr is positive
        or None to disable stopping. context supplies live SoC/device handles,
        cancellation and plots. req is unused.
        Return this run's typed cfg and result. Acquisition errors propagate.
        """
        del req
        result = PowerDepExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)
