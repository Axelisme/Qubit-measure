from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    AnalyzeResultBase,
    NoAnalyzeParams,
    RunRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.singleshot_shared import (
    read_confusion_matrix,
)
from zcu_lab.v2.singleshot.mist.freq.core import (
    FreqAnalyzeOptions,
    FreqCfg,
    FreqDepExp,
    FreqResult,
)
from zcu_lab.v2.singleshot.mist.freq.form import make_form

MistFreqRunResult: TypeAlias = RunRecord[FreqCfg, FreqResult]


@dataclass
class MistFreqAnalyzeResult(AnalyzeResultBase):
    pass


class MistFreqAdapter(
    BaseAdapter[FreqCfg, MistFreqRunResult, MistFreqAnalyzeResult, NoAnalyzeParams]
):
    experiment_tag: ClassVar[str | None] = FreqDepExp.persistence_tag()

    @staticmethod
    def make_form(ctx: FormContext) -> Form[FreqCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    exp_cls = FreqDepExp

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    # No get_analyze_params override: NoAnalyzeParams (4th generic arg).

    def run(
        self,
        req: RunRequest,
        cfg: FreqCfg,
        *,
        context: RunContext,
    ) -> MistFreqRunResult:
        del req
        return RunRecord(cfg, FreqDepExp().run(cfg, context=context))

    def analyze(
        self,
        req: AnalyzeRequest[MistFreqRunResult, NoAnalyzeParams],
        *,
        plots: Plots,
    ) -> MistFreqAnalyzeResult:
        options = FreqAnalyzeOptions(
            confusion_matrix=read_confusion_matrix(req.parameters),
        )
        FreqDepExp().analyze(req.run_result, options, plots=plots)
        return MistFreqAnalyzeResult()
