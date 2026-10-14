from __future__ import annotations

import time
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
    SessionEnv,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.twotone.reset.dual_tone.length.core import (
    LengthCfg,
    LengthExp,
    LengthResult,
)
from zcu_lab.v2.twotone.reset.dual_tone.length.form import make_form

DualToneLengthRunResult: TypeAlias = RunRecord[LengthCfg, LengthResult]


@dataclass
class DualToneLengthAnalyzeResult(AnalyzeResultBase):
    """Length analysis publishes a curve without fitted scalar output."""

    pass


class DualToneLengthAdapter(
    BaseAdapter[
        LengthCfg,
        DualToneLengthRunResult,
        DualToneLengthAnalyzeResult,
        NoAnalyzeParams,
    ]
):
    experiment_tag: ClassVar[str | None] = LengthExp.persistence_tag()
    exp_cls = LengthExp

    @staticmethod
    def make_form(ctx: FormContext) -> Form[LengthCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    # No get_analyze_params override: NoAnalyzeParams (the 4th generic arg) makes
    # BaseAdapter return the empty params instance and reflect the type.

    def run(
        self,
        req: RunRequest,
        cfg: LengthCfg,
        *,
        context: RunContext,
    ) -> DualToneLengthRunResult:
        del req
        result = LengthExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self,
        req: AnalyzeRequest[DualToneLengthRunResult, NoAnalyzeParams],
        *,
        plots: Plots,
    ) -> DualToneLengthAnalyzeResult:
        LengthExp().analyze(req.run_result, None, plots=plots)
        return DualToneLengthAnalyzeResult()

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_dualreset_length_{time.strftime('%m%d')}"
