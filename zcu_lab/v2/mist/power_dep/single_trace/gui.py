"""Measure-GUI adapter for a MIST gain trace with averaged complex IQ."""

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

from zcu_lab.v2.mist.power_dep.single_trace.core import (
    PowerDepAnalyzeOptions,
    PowerDepCfg,
    PowerDepExp,
    PowerDepResult,
)
from zcu_lab.v2.mist.power_dep.single_trace.form import make_form

MistTraceRunResult: TypeAlias = RunRecord[PowerDepCfg, PowerDepResult]


@dataclass
class MistTraceAnalyzeParams:
    """IQ references as Python complex strings (e.g. 1+2j), or None to omit.

    g0=None uses the complex mean of the first 5% of scan points (at least one).
    e0 only sets the upper display limit, at 1.1 times its distance from the reference.
    ac_coeff=None keeps gain on the x-axis; a supplied coefficient maps gain
    squared to photon number. Invalid complex strings fail analysis.
    """

    g0: Annotated[
        str | None, ParamMeta(label="Ground IQ reference (blank = scan start mean)")
    ] = None
    e0: Annotated[str | None, ParamMeta(label="Excited IQ (optional y-limit)")] = None
    ac_coeff: Annotated[
        float | None, ParamMeta(label="Photon number / gain squared", decimals=6)
    ] = None


def _analyze(
    req: AnalyzeRequest[MistTraceRunResult, MistTraceAnalyzeParams],
    *,
    plots: Plots,
) -> None:
    """Plot source IQ displacement using the submitted reference/axis options.

    Accept loaded sources without cfg. Invalid complex reference strings
    raise ValueError; analysis does not read the current measurement form.
    """
    params = req.analyze_params
    options = PowerDepAnalyzeOptions(
        g0=None if params.g0 is None else complex(params.g0),
        e0=None if params.e0 is None else complex(params.e0),
        ac_coeff=params.ac_coeff,
    )
    PowerDepExp().analyze(req.run_result, options, plots=plots)


def _summary_lines(source: MistTraceRunResult) -> tuple[str, ...]:
    cfg = source.cfg
    return () if cfg is None else (f"Reps × Rounds: {cfg.reps} × {cfg.rounds}",)


def run(
    cfg: PowerDepCfg, *, context: RunContext
) -> RunRecord[PowerDepCfg, PowerDepResult]:
    """Acquire with cfg and retain it as the analysis and save source."""
    return RunRecord(cfg=cfg, result=PowerDepExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=PowerDepExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_analyze,
        options=lambda source, env: MistTraceAnalyzeParams(),
        summary_lines=_summary_lines,
    ),
)
