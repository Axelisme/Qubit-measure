"""Measure GUI for gain-resolved relaxation during a fast-flux pulse."""

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TypeAlias

import numpy as np
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    AnalyzeRequest,
    NoAnalyzeParams,
    RunRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from .core import T1Analysis, T1Cfg, T1Exp, T1Result
from .form import make_form

T1RunResult: TypeAlias = RunRecord[T1Cfg, T1Result]


@dataclass(frozen=True)
class T1AnalyzeResult:
    """Gain-resolved decay estimates; rejected fits are null in the summary."""

    gains: list[float]
    analysis: T1Analysis

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "gains": self.gains,
            "t1_us": [
                float(value) if np.isfinite(value) else None
                for value in self.analysis.t1s
            ],
            "t1_err_us": [
                float(error) if np.isfinite(value) and np.isfinite(error) else None
                for value, error in zip(
                    self.analysis.t1s, self.analysis.t1errs, strict=True
                )
            ],
        }


class T1Adapter(BaseAdapter[T1Cfg, T1RunResult, T1AnalyzeResult, NoAnalyzeParams]):
    """Delegate the physical sequence, fit and native persistence to the core."""

    exp_cls = T1Exp
    experiment_tag: ClassVar[str | None] = T1Exp.persistence_tag()
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.FIT, load_data=True
    )
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @staticmethod
    def make_form(ctx: FormContext) -> Form[T1Cfg]:
        return make_form(ctx)

    def run(self, req: RunRequest, cfg: T1Cfg, *, context: RunContext) -> T1RunResult:
        del req
        return RunRecord(cfg=cfg, result=T1Exp().run(cfg, context=context))

    def analyze(
        self, req: AnalyzeRequest[T1RunResult, NoAnalyzeParams], *, plots: Plots
    ) -> T1AnalyzeResult:
        source = req.run_result
        return T1AnalyzeResult(
            gains=source.result.gains.tolist(),
            analysis=T1Exp().analyze(source, None, plots=plots),
        )
