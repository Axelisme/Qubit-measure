from __future__ import annotations

from pathlib import Path
from typing import ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    RunRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.singleshot_shared import (
    read_confusion_matrix,
)
from zcu_lab.v2.singleshot.mist.power_freq.core import (
    FreqPowerAnalyzeOptions,
    FreqPowerCfg,
    FreqPowerExp,
    FreqPowerResult,
)
from zcu_lab.v2.singleshot.mist.power_freq.form import make_form

MistPowerFreqRunResult: TypeAlias = RunRecord[FreqPowerCfg, FreqPowerResult]


class MistPowerFreqAdapter(BaseAdapter[FreqPowerCfg, MistPowerFreqRunResult]):
    @staticmethod
    def make_form(ctx: FormContext) -> Form[FreqPowerCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    exp_cls = FreqPowerExp

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    # No get_analyze_params override: None (4th generic arg).

    def run(
        self,
        req: RunRequest,
        cfg: FreqPowerCfg,
        *,
        context: RunContext,
    ) -> MistPowerFreqRunResult:
        del req
        return RunRecord(cfg, FreqPowerExp().run(cfg, context=context))

    def analyze(
        self,
        req: AnalyzeRequest[MistPowerFreqRunResult, None],
        *,
        plots: Plots,
    ) -> None:
        options = FreqPowerAnalyzeOptions(
            confusion_matrix=read_confusion_matrix(req.parameters),
        )
        FreqPowerExp().analyze(req.run_result, options, plots=plots)
        return
