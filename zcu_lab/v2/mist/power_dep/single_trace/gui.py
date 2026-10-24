"""Measure-GUI adapter for a MIST gain trace with averaged complex IQ."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, TypeAlias

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

    g0=None uses the first measured IQ. e0 only sets the vertical display range.
    ac_coeff=None keeps gain on the x-axis; a supplied coefficient maps gain
    squared to photon number. Invalid complex strings fail analysis.
    """

    g0: Annotated[str | None, ParamMeta(label="Ground IQ (empty = first point)")] = None
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
    return


DEFINITION = Adapter(
    spec=PowerDepExp.AXES_SPEC,
    make_form=make_form,
    run=PowerDepExp().run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_analyze, options=lambda source, env: MistTraceAnalyzeParams()
    ),
)
