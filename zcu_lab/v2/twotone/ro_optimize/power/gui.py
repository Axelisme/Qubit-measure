from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar, Literal, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    ParameterWriteback,
    ParamMeta,
    RunRequest,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.writeback_helpers import readout_dpm_writeback_items
from zcu_lab.v2.twotone.ro_optimize.power.core import (
    PowerAnalysis,
    PowerAnalyzeOptions,
    PowerCfg,
    PowerExp,
    PowerResult,
)
from zcu_lab.v2.twotone.ro_optimize.power.form import make_form

RoOptPowerRunResult: TypeAlias = RunRecord[PowerCfg, PowerResult]


@dataclass
class RoOptPowerAnalyzeParams:
    smooth_method: Annotated[
        Literal["wavelet", "gaussian"], ParamMeta(label="Smooth method")
    ] = "wavelet"
    smooth: Annotated[float, ParamMeta(label="Smooth strength", decimals=2)] = 1.0
    penalty_ratio: Annotated[
        float, ParamMeta(label="Power penalty ratio", decimals=2)
    ] = 0.5


class RoOptPowerAdapter(
    BaseAdapter[
        PowerCfg,
        RoOptPowerRunResult,
        PowerAnalysis,
        RoOptPowerAnalyzeParams,
    ]
):
    exp_cls = PowerExp

    @staticmethod
    def make_form(ctx: FormContext) -> Form[PowerCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    def run(
        self,
        req: RunRequest,
        cfg: PowerCfg,
        *,
        context: RunContext,
    ) -> RoOptPowerRunResult:
        del req
        return RunRecord(cfg, PowerExp().run(cfg, context=context))

    def analyze(
        self,
        req: AnalyzeRequest[RoOptPowerRunResult, RoOptPowerAnalyzeParams],
        *,
        plots: Plots,
    ) -> PowerAnalysis:
        params = req.analyze_params
        options = PowerAnalyzeOptions(
            penalty_ratio=params.penalty_ratio,
            smooth=params.smooth,
            smooth_method=params.smooth_method,
        )
        return PowerExp().analyze(req.run_result, options, plots=plots)

    def get_writeback_items(
        self, req: WritebackRequest[RoOptPowerRunResult, PowerAnalysis]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        items: list[WritebackItem] = [
            ParameterWriteback(
                target_name="$qubit.ext.best_ro_gain",
                description="Optimal readout gain (a.u.)",
                proposed_value=result.best_gain,
            ),
        ]
        items.extend(
            readout_dpm_writeback_items(
                req.ctx,
                req.run_result.cfg,
                proposed={"best_ro_gain": result.best_gain},
            )
        )
        return items
