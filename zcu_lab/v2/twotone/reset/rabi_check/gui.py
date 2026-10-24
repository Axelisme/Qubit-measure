from __future__ import annotations

from pathlib import Path
from typing import ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    AnalyzeRequest,
    RunRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.twotone.reset.rabi_check.core import (
    RabiCheckCfg,
    RabiCheckExp,
    RabiCheckResult,
)
from zcu_lab.v2.twotone.reset.rabi_check.fit import RabiCheckFit
from zcu_lab.v2.twotone.reset.rabi_check.form import make_form

RabiCheckRunResult: TypeAlias = RunRecord[RabiCheckCfg, RabiCheckResult]


class RabiCheckAdapter(BaseAdapter[RabiCheckCfg, RabiCheckRunResult, RabiCheckFit]):
    """Rabi-amplitude check for any reset type (single-tone / two-pulse / bath).

    Sweeps the initialisation pulse gain and records three signal branches in
    parallel (without tested_reset / with tested_reset / with tested_reset + rabi_pulse),
    fitting relative Rabi contrast and residual input dependence. Averaged IQ
    diagnostics do not determine reset fidelity or a unique reset channel.
    """

    exp_cls = RabiCheckExp

    @staticmethod
    def make_form(ctx: FormContext) -> Form[RabiCheckCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        requires_soc=True, analysis=AnalysisMode.FIT, load_data=True
    )

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    def run(
        self,
        req: RunRequest,
        cfg: RabiCheckCfg,
        *,
        context: RunContext,
    ) -> RabiCheckRunResult:
        del req
        result = RabiCheckExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self, req: AnalyzeRequest[RabiCheckRunResult, None], *, plots: Plots
    ) -> RabiCheckFit:
        return RabiCheckExp().analyze(req.run_result, None, plots=plots)
