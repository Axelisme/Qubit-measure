from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    ParameterWriteback,
    ParamMeta,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.support.measure.writeback_helpers import readout_dpm_writeback_items
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


def _RoOptPower_analyze(
    req: AnalyzeRequest[RoOptPowerRunResult, RoOptPowerAnalyzeParams], *, plots: Plots
) -> PowerAnalysis:
    params = req.analyze_params
    options = PowerAnalyzeOptions(
        penalty_ratio=params.penalty_ratio,
        smooth=params.smooth,
        smooth_method=params.smooth_method,
    )
    return PowerExp().analyze(req.run_result, options, plots=plots)


def _RoOptPower_get_writeback_items(
    req: WritebackRequest[RoOptPowerRunResult, PowerAnalysis],
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


def _RoOptPower_make_form(ctx: FormContext) -> Form[PowerCfg]:
    """Build this adapter's definition using captured Open/Reset sources."""
    return make_form(ctx)


def _run(cfg: PowerCfg, *, context: RunContext) -> RunRecord[PowerCfg, PowerResult]:
    """Acquire a trace and retain the experiment configuration with its result."""
    return RunRecord(cfg=cfg, result=PowerExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=PowerExp.AXES_SPEC,
    make_form=_RoOptPower_make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_RoOptPower_analyze,
        options=lambda source, env: RoOptPowerAnalyzeParams(),
        writeback=_RoOptPower_get_writeback_items,
    ),
)
