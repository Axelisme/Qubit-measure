from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    AnalyzeResultBase,
    ParameterWriteback,
    RunRequest,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.writeback_helpers import readout_dpm_writeback_items
from zcu_lab.v2.twotone.ro_optimize.auto_optimize.core import (
    AutoOptCfg,
    AutoOptExp,
    AutoOptResult,
)
from zcu_lab.v2.twotone.ro_optimize.auto_optimize.form import make_form

RoOptAutoRunResult: TypeAlias = RunRecord[AutoOptCfg, AutoOptResult]


@dataclass
class RoOptAutoAnalyzeResult(AnalyzeResultBase):
    best_freq: float
    best_gain: float
    best_length: float


class RoOptAutoAdapter(
    BaseAdapter[AutoOptCfg, RoOptAutoRunResult, RoOptAutoAnalyzeResult]
):
    experiment_tag: ClassVar[str | None] = AutoOptExp.persistence_tag()
    exp_cls = AutoOptExp

    @staticmethod
    def make_form(ctx: FormContext) -> Form[AutoOptCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    def run(
        self,
        req: RunRequest,
        cfg: AutoOptCfg,
        *,
        context: RunContext,
    ) -> RoOptAutoRunResult:
        del req
        return RunRecord(cfg, AutoOptExp().run(cfg, context=context))

    def analyze(
        self,
        req: AnalyzeRequest[RoOptAutoRunResult, None],
        *,
        plots: Plots,
    ) -> RoOptAutoAnalyzeResult:
        result = AutoOptExp().analyze(req.run_result, None, plots=plots)
        return RoOptAutoAnalyzeResult(
            best_freq=result.best_freq,
            best_gain=result.best_gain,
            best_length=result.best_length,
        )

    def get_writeback_items(
        self, req: WritebackRequest[RoOptAutoRunResult, RoOptAutoAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        items: list[WritebackItem] = [
            ParameterWriteback(
                target_name="$qubit.ext.best_ro_freq",
                description="Optimal readout frequency (MHz)",
                proposed_value=result.best_freq,
            ),
            ParameterWriteback(
                target_name="$qubit.ext.best_ro_gain",
                description="Optimal readout gain (a.u.)",
                proposed_value=result.best_gain,
            ),
            ParameterWriteback(
                target_name="$qubit.ext.best_ro_length",
                description="Optimal readout length (us)",
                proposed_value=result.best_length,
            ),
        ]
        items.extend(
            readout_dpm_writeback_items(
                req.ctx,
                req.run_result.cfg,
                proposed={
                    "best_ro_freq": result.best_freq,
                    "best_ro_gain": result.best_gain,
                    "best_ro_length": result.best_length,
                },
            )
        )
        return items
