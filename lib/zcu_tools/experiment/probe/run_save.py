"""Independent acquisition and native persistence stages for offline callers."""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Generic, TypeVar
from urllib.parse import quote

import numpy as np
from matplotlib import rc_context, style

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.datafile import load
from zcu_tools.plotting import DEFAULT_STYLE
from zcu_tools.plotting.plots import NonPresentingHost, Plots

from .. import native_persistence
from ..axes_spec import AxesSpec, GroupedAxesSpec
from ..base import Experiment
from ..records import RunRecord
from ..simulation import SimulationEnvironment
from ._acquisition import acquire_record
from ._comparison import compare_roundtrip

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


@dataclass(frozen=True)
class SaveStageResult(Generic[_Cfg, _Result]):
    """Native persistence output with a field-by-field Result comparison.

    record is the restored RunRecord, not the saving input. native_path is the
    written directory/data.h5. roundtrip_equal follows compare_roundtrip's
    shape, dtype, exact-field and scaled-field rules; False reports a mismatch.
    cfg_issue is None for restored cfg or the native owner's diagnostic when
    historical cfg cannot be reconstructed (record.cfg is then None).
    """

    record: RunRecord[_Cfg, _Result]
    native_path: Path
    roundtrip_equal: bool
    cfg_issue: str | None


def save_stage(
    record: RunRecord[_Cfg, _Result],
    spec: AxesSpec[_Result, _Cfg] | GroupedAxesSpec[_Result, _Cfg],
    directory: Path,
) -> SaveStageResult[_Cfg, _Result]:
    """Save to directory/data.h5, restore and compare every Result field.

    Accept a directly constructed record or run_stage's record. cfg and Result
    must match spec; metadata must satisfy native RunMetadata and its tag.
    Use native_persistence for all encoding/restoration, create directories and
    atomically replace an existing data.h5. No environment or analysis is used.
    Return the restored record and cfg diagnostic, path and compare_roundtrip's
    boolean comparison, including its scaled-field tolerance and grouped rules.
    A differing field returns False, not an exception. Validation, I/O, restore
    and comparison errors propagate unchanged; an already saved file remains.
    """
    native_path = directory / "data.h5"
    native_persistence.save_run(record, native_path, spec=spec, replace=True)
    restored = native_persistence.restore_run(load(native_path), spec=spec)
    return SaveStageResult(
        restored.record,
        native_path,
        compare_roundtrip(record.result, restored.record.result, spec=spec),
        restored.cfg_issue,
    )


def _validate_dtypes(
    result: _Result, spec: AxesSpec[_Result, _Cfg] | GroupedAxesSpec[_Result, _Cfg]
) -> None:
    declarations = (
        (*spec.axes, spec.z)
        if isinstance(spec, AxesSpec)
        else tuple(item for variable in spec.variables for item in (*variable.axes, variable.z))
    )
    for declaration in declarations:
        if declaration.field_name is None:
            continue
        actual = np.asarray(getattr(result, declaration.field_name)).dtype
        expected = np.dtype(declaration.dtype)
        if actual != expected:
            raise ValueError(
                f"Result field {declaration.field_name!r} dtype {actual} != declared {expected}"
            )


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
    validation raises ValueError. Field-backed axes and signal arrays must have
    the exact declared NumPy memory dtype; mismatches raise ValueError naming
    the field. This stage neither saves nor calls analyze.
    """
    if experiment.AXES_SPEC is None:
        raise ValueError("run_stage requires experiment.AXES_SPEC")
    started = perf_counter()
    style.use(DEFAULT_STYLE)
    plots = Plots(NonPresentingHost())
    try:
        record = acquire_record(experiment, cfg, environment, plots)
        _validate_dtypes(record.result, experiment.AXES_SPEC)
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
