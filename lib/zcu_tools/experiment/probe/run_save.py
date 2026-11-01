"""Independent acquisition and native persistence stages for offline callers."""

from dataclasses import dataclass
from io import BytesIO
from time import perf_counter
from typing import Generic, TypeVar
from urllib.parse import quote

from matplotlib import rc_context, style

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.plotting import DEFAULT_STYLE
from zcu_tools.plotting.plots import NonPresentingHost, Plots

from .. import native_persistence
from ..base import Experiment
from ..records import RunRecord
from ..simulation import SimulationEnvironment

_Cfg = TypeVar("_Cfg", bound=ConfigBase)
_Result = TypeVar("_Result")


@dataclass(frozen=True)
class RunStageResult(Generic[_Cfg, _Result]):
    """Acquisition output without persistence or analysis.

    record holds the isolated input cfg, acquired Result and native run metadata.
    elapsed_seconds is wall time for acquisition and PNG rendering in seconds.
    pngs maps live-{URL-escaped figure name}.png to Agg PNG bytes at the figure's
    current size and 100 dpi. An experiment without figures returns an empty map.
    """

    record: RunRecord[_Cfg, _Result]
    elapsed_seconds: float
    pngs: dict[str, bytes]


def run_stage(
    experiment: Experiment[_Result, _Cfg],
    cfg: _Cfg,
    environment: SimulationEnvironment,
) -> RunStageResult[_Cfg, _Result]:
    """Acquire a record and render its live figures without writing files.

    cfg must match experiment.AXES_SPEC.cfg_type. The experiment receives a deep
    copy and a fresh run Scope/Plots using only the caller's environment handles.
    Caller owns environment cleanup; this stage never closes or rewires it.
    Apply DEFAULT_STYLE to process-wide rcParams before run. Agg canvases and a
    NonPresentingHost update figures without opening views. Return PNG bytes at
    current figure size, 100 dpi, under live-{URL-escaped name}.png keys.
    Missing AXES_SPEC raises ValueError. Acquisition, plotting and rendering
    errors propagate unchanged. Result shape must agree with AXES_SPEC or native
    validation raises ValueError. This stage neither saves nor calls analyze.
    """
    # The package defines the existing acquisition helper after its export imports.
    from . import _acquire

    if experiment.AXES_SPEC is None:
        raise ValueError("run_stage requires experiment.AXES_SPEC")
    started = perf_counter()
    style.use(DEFAULT_STYLE)
    plots = Plots(NonPresentingHost())
    try:
        record = _acquire(experiment, cfg, environment, plots)
        native_persistence.prepare_run_data(record, spec=experiment.AXES_SPEC, comment="")
        plots.finish()
        pngs: dict[str, bytes] = {}
        for name, figure in plots.items():
            if figure.get_layout_engine() is None:
                figure.set_layout_engine("tight")
            with BytesIO() as buffer, rc_context({"savefig.bbox": None}):
                figure.savefig(buffer, format="png", dpi=100)
                pngs[f"live-{quote(name, safe='')}.png"] = buffer.getvalue()
        return RunStageResult(record, perf_counter() - started, pngs)
    finally:
        plots.finish(present=False)
        plots.release()
