"""Measure GUI adapter for the fast-flux MIST contrast map."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    ParamMeta,
)
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.fastflux.mist.core import (
    MistAnalyzeOptions,
    MistCfg,
    MistExp,
    MistResult,
)
from zcu_lab.v2.fastflux.mist.form import make_form

MistRunResult: TypeAlias = RunRecord[MistCfg, MistResult]


@dataclass
class MistAnalyzeParams:
    ac_coeff: Annotated[
        float | None, ParamMeta(label="Photon number / gain squared")
    ] = None


def _analyze(
    req: AnalyzeRequest[MistRunResult, MistAnalyzeParams],
    *,
    plots: Plots,
) -> None:
    """Plot the stored map, optionally scaling gain squared to photons."""
    MistExp().analyze(
        req.run_result,
        MistAnalyzeOptions(ac_coeff=req.analyze_params.ac_coeff),
        plots=plots,
    )
    return


def run(cfg: MistCfg, *, context: RunContext) -> RunRecord[MistCfg, MistResult]:
    """Acquire with cfg and retain it as the analysis and save source."""
    return RunRecord(cfg=cfg, result=MistExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=MistExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze, options=lambda source, env: MistAnalyzeParams()),
)
