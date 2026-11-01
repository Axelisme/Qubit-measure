"""Analyze caller records without running acquisition or persistence."""

from dataclasses import dataclass
from io import BytesIO
from typing import Any, TypeVar
from urllib.parse import quote

from matplotlib import rc_context, style

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.plotting import DEFAULT_STYLE
from zcu_tools.plotting.plots import NonPresentingHost, Plots

from ..base import Experiment
from ..records import RunRecord

_Cfg = TypeVar("_Cfg", bound=ConfigBase)
_Result = TypeVar("_Result")


@dataclass(frozen=True)
class AnalysisStageResult:
    """One completed analysis and its exported figures.

    analysis is the actual return value of Experiment.analyze, not a summary.
    pngs maps analysis-{URL-escaped figure name}.png to PNG bytes at the
    figure's current size and 100 dpi. An analysis without figures has {}.
    """

    analysis: Any
    pngs: dict[str, bytes]


def analysis_stage(
    experiment: Experiment[_Result, _Cfg],
    record: RunRecord[_Cfg, _Result],
    options: Any,
) -> AnalysisStageResult:
    """Analyze an explicit record with experiment-specific options.

    No acquisition, hardware, AXES_SPEC or persistence is required. Pass options
    as accepted by experiment.analyze; its return value is retained unchanged.
    Use Agg figures and a NonPresentingHost, applying the shared style before
    analyze. No files or temporary directories are created. Caller decides
    where to save returned PNG bytes. All analyze/rendering errors propagate.
    Presentation is released on success and failure.
    """
    style.use(DEFAULT_STYLE)
    plots = Plots(NonPresentingHost())
    try:
        analysis = experiment.analyze(record, options, plots=plots)
        plots.finish()
        pngs: dict[str, bytes] = {}
        for name in plots:
            buffer = BytesIO()
            with rc_context({"savefig.bbox": None}):
                plots[name].savefig(buffer, format="png", dpi=100)
            pngs[f"analysis-{quote(name, safe='')}.png"] = buffer.getvalue()
        return AnalysisStageResult(analysis, pngs)
    finally:
        plots.finish(present=False)
        plots.release()
