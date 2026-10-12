"""Committed T1 raw decay policy for terminal workflow artifacts."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from pydantic import TypeAdapter
from zcu_tools.datafile import (
    Axis,
    CfgSnapshot,
    DataVariable,
    ExperimentPayload,
    LabberMetadata,
    LabberPayload,
    VariableSchema,
    load_run_data,
    validate_experiment_payload,
)
from zcu_tools.experiment.axes_spec import AxesSpec
from zcu_tools.experiment.native_persistence import native_schemas
from zcu_tools.experiment.workflows import CommittedRecord
from zcu_tools.experiment.workflows.exporting import PreparedFluxdepExport

from zcu_lab.v2.mist.power_dep.single_trace.core import PowerDepExp
from zcu_lab.v2.twotone.time_domain.t1.core import T1Exp
from zcu_lab.v2.twotone.time_domain.t2echo.core import T2EchoExp
from zcu_lab.v2.twotone.time_domain.t2ramsey.core import T2RamseyExp

from .mist import MISTPlan, MISTPoint
from .samples import export_t1_samples, export_t2echo_samples, export_t2ramsey_samples

__all__ = [
    "export_t1_samples",
    "export_t2echo_samples",
    "export_t2ramsey_samples",
    "prepare_t1_fluxdep_export",
    "prepare_t2echo_fluxdep_export",
    "prepare_t2ramsey_fluxdep_export",
    "prepare_mist_fluxdep_export",
]
from .t1 import T1Plan, T1Point
from .t2 import T2Plan, T2Point

_POINT = TypeAdapter(T1Point)
_T2_POINT = TypeAdapter(T2Point)
_MIST_POINT = TypeAdapter(MISTPoint)


def prepare_t1_fluxdep_export(
    *, plan: T1Plan, data_root: Path, records: Sequence[CommittedRecord]
) -> PreparedFluxdepExport:
    """Prepare one detached payload without writing or fitting measurements.

    plan is the fixed native Start plan, including absolute setpoints/flux_unit
    and historical snapshot. data_root is the host's exact acquisition root.
    records must come from Engine.records for this run; record.json is never
    consulted. Only final T1 acquisitions whose native metadata matches the T1
    persistence declaration are used, not spectroscopy/Rabi/readout calibration.
    Each record's index selects its plan slot and flux supplies its actual
    setpoint. Missing slots and shorter traces are NaN-filled. Signal is complex
    a.u.; Time contains actual seconds, both flux-by-index. Time Index has no
    physical unit and spans the longest usable committed trace. T1 is a Flux-only
    seconds vector; rejected fits remain NaN while their raw trace is retained.
    Different delays are never unioned, rounded or interpolated.

    Return the same ExperimentPayload for native/independent Labber writers and
    a complete T1Plan cfg snapshot. No usable committed acquisition, degraded or
    malformed point summaries, duplicate/out-of-range indices, escaped file
    paths, invalid native T1 data, or invalid point file references raise
    ValueError before export writes. Reader I/O errors propagate. Uncommitted
    files, even if published on disk, are not examined.
    """
    traces: dict[int, LabberPayload] = {}
    fluxes = np.array(plan.fluxes, dtype=float)
    lifetimes = np.full(len(fluxes), np.nan, dtype=float)
    seen: set[int] = set()
    root = data_root.resolve()
    for record in records:
        if record.encoded_record.mode != "json":
            raise ValueError("T1 export requires structured committed point summaries")
        point = _POINT.validate_python(record.encoded_record.value)
        if point.index < 0 or point.index >= len(fluxes) or point.index in seen:
            raise ValueError("T1 committed index must be unique and inside the plan")
        seen.add(point.index)
        fluxes[point.index] = point.flux
        trace = _load_final_t1(root, record, point.decay_file)
        if trace is not None:
            traces[point.index] = trace
        if point.index in traces and point.lifetime_us is not None:
            if not np.isfinite(point.lifetime_us) or point.lifetime_us <= 0:
                raise ValueError("accepted T1 lifetime must be finite and positive")
            lifetimes[point.index] = point.lifetime_us / 1e6
    if not traces:
        raise ValueError("No usable committed T1 acquisition")
    samples = max(len(trace.z) for trace in traces.values())
    signal = np.full((len(fluxes), samples), complex(np.nan, np.nan), dtype=complex)
    time = np.full(signal.shape, np.nan, dtype=float)
    for index, trace in traces.items():
        count = len(trace.z)
        signal[index, :count] = trace.z
        time[index, :count] = trace.axes[0].values
    flux_axis = Axis("Flux", plan.flux_unit, fluxes)
    axes = [Axis("Time Index", "", np.arange(samples)), flux_axis]
    payload = ExperimentPayload(
        variables={
            DataVariable("signal"): LabberPayload(("Signal", "a.u.", signal), axes),
            DataVariable("time"): LabberPayload(("Time", "s", time), axes),
            DataVariable("t1"): LabberPayload(("T1", "s", lifetimes), [flux_axis]),
        },
        metadata=LabberMetadata(
            comment="Committed T1 fluxdep raw decay and accepted lifetime"
        ),
        representation="grouped",
    )
    return PreparedFluxdepExport(
        payload,
        CfgSnapshot(
            values=plan.model_dump(mode="json"), cfg_type="T1Plan", schema_version="1.0"
        ),
    )


def _load_final_t1(
    root: Path, record: CommittedRecord, filename: str | None
) -> LabberPayload | None:
    """Select only the declared final decay from this committed invocation."""
    spec = T1Exp().persistence_spec()
    if not isinstance(spec, AxesSpec):
        raise ValueError("Final T1 must declare a single-variable acquisition")
    return _load_declared_trace(
        root, record, filename, spec.tag, native_schemas(spec)[0]
    )


def _load_declared_trace(
    root: Path,
    record: CommittedRecord,
    filename: str | None,
    tag: str,
    schema: VariableSchema,
) -> LabberPayload | None:
    if filename is None:
        return None
    path = (root / record.iteration_dir / filename).resolve()
    if not path.is_relative_to(root):
        raise ValueError("committed acquisition path escapes data_root")
    stored = load_run_data(path)
    if stored.metadata.experiment != tag:
        return None
    validate_experiment_payload(stored.payload, schema=(schema,))
    trace = stored.payload.variables[schema.variable]
    return trace if len(trace.z) else None


def prepare_t2echo_fluxdep_export(
    *, plan: T2Plan, data_root: Path, records: Sequence[CommittedRecord]
) -> PreparedFluxdepExport:
    """Prepare committed final Echo Signal/Time 2D and Flux-only T2echo seconds.

    Supply the fixed Start T2Plan, exact acquisition root and Engine.records.
    Native twotone/ge/t2echo declaration identifies only final traces. Index
    selects the plan slot, flux is the actual committed setpoint and lifetime
    is the committed accepted scalar, never refitted. Short/missing rows and
    rejected lifetimes remain NaN. Inner-first axes are Time Index and Flux;
    delays retain actual seconds without union, rounding or interpolation.
    Return one grouped payload and full plan cfg for both writers. Malformed
    summaries/indices, escaped paths, invalid named traces or no usable
    committed trace raise ValueError before writing; reader errors propagate.
    No terminal admission, journal/file-summary read, fitting or I/O write occurs.
    """
    return _prepare_t2(plan, data_root, records, echo=True)


def prepare_t2ramsey_fluxdep_export(
    *, plan: T2Plan, data_root: Path, records: Sequence[CommittedRecord]
) -> PreparedFluxdepExport:
    """Prepare committed final Ramsey Signal/Time 2D and T2ramsey 1D seconds.

    Inputs, axes, sparse/rejected-row handling and failures follow
    prepare_t2echo_fluxdep_export, using only native twotone/ge/t2ramsey traces.
    The committed lifetime is authoritative; no detune/calibration refit or
    live state lookup occurs. Return the same detached payload/cfg for both
    explicit writers. No host admission, path allocation or write occurs.
    """
    return _prepare_t2(plan, data_root, records, echo=False)


def _prepare_t2(
    plan: T2Plan, data_root: Path, records: Sequence[CommittedRecord], *, echo: bool
) -> PreparedFluxdepExport:
    if echo:
        echo_spec = T2EchoExp().persistence_spec()
        tag, schemas = echo_spec.tag, native_schemas(echo_spec)
    else:
        ramsey_spec = T2RamseyExp().persistence_spec()
        tag, schemas = ramsey_spec.tag, native_schemas(ramsey_spec)
    if len(schemas) != 1:
        raise ValueError("Final T2 must declare a single-variable acquisition")
    label = "T2echo" if echo else "T2ramsey"
    traces: dict[int, LabberPayload] = {}
    fluxes = np.array(plan.fluxes, dtype=float)
    lifetimes = np.full(len(fluxes), np.nan)
    seen: set[int] = set()
    for record in records:
        if record.encoded_record.mode != "json":
            raise ValueError("T2 export requires structured committed point summaries")
        point = _T2_POINT.validate_python(record.encoded_record.value)
        if point.index < 0 or point.index >= len(fluxes) or point.index in seen:
            raise ValueError("T2 committed index must be unique and inside the plan")
        seen.add(point.index)
        fluxes[point.index] = point.flux
        trace = _load_declared_trace(
            data_root.resolve(), record, point.decay_file, tag, schemas[0]
        )
        if trace is not None:
            traces[point.index] = trace
            if point.lifetime_us is not None:
                lifetimes[point.index] = point.lifetime_us / 1e6
    if not traces:
        raise ValueError("No usable committed T2 acquisition")
    samples = max(len(trace.z) for trace in traces.values())
    signal = np.full((len(fluxes), samples), complex(np.nan, np.nan))
    time = np.full(signal.shape, np.nan)
    for index, trace in traces.items():
        count = len(trace.z)
        signal[index, :count] = trace.z
        time[index, :count] = trace.axes[0].values
    flux_axis = Axis("Flux", plan.flux_unit, fluxes)
    axes = [Axis("Time Index", "", np.arange(samples)), flux_axis]
    return PreparedFluxdepExport(
        ExperimentPayload(
            variables={
                DataVariable("signal"): LabberPayload(("Signal", "a.u.", signal), axes),
                DataVariable("time"): LabberPayload(("Time", "s", time), axes),
                DataVariable(label.lower()): LabberPayload(
                    (label, "s", lifetimes), [flux_axis]
                ),
            },
            metadata=LabberMetadata(
                comment=f"Committed {label} raw decay and accepted lifetime"
            ),
            representation="grouped",
        ),
        CfgSnapshot(
            values=plan.model_dump(mode="json"), cfg_type="T2Plan", schema_version="1.0"
        ),
    )


def prepare_mist_fluxdep_export(
    *, plan: MISTPlan, data_root: Path, records: Sequence[CommittedRecord]
) -> PreparedFluxdepExport:
    """Prepare raw MIST Signal Flux-by-Readout-Gain and Flux-only Success.

    plan is the fixed Start MISTPlan with run-locked gain start/stop/points;
    data_root is the exact root and records must be Engine.records. Only native
    mist/gain final traces are read, not calibration, journal or record.json.
    Inner-first axes are actual Readout Gain (a.u.) and Flux (plan.flux_unit).
    Success is dimensionless float 1.0 for committed successful acquisition,
    otherwise NaN; it is not a fitted threshold. Missing rows are NaN-filled.
    Any native gain grid unequal to the exact fixed plan grid raises ValueError,
    without rounding, tolerance repair, union or interpolation. Invalid summary,
    index/path, invalid named trace or no usable committed acquisition also raises;
    reader errors propagate. Return a shared grouped payload and full plan cfg,
    with no invented Time/lifetime channels, host admission or writes.
    """
    gains = np.linspace(plan.gain_start, plan.gain_stop, plan.gain_points)
    fluxes = np.array(plan.fluxes, dtype=float)
    signal = np.full((len(fluxes), len(gains)), complex(np.nan, np.nan))
    success = np.full(len(fluxes), np.nan)
    seen: set[int] = set()
    usable = False
    spec = PowerDepExp().persistence_spec()
    schemas = native_schemas(spec)
    if len(schemas) != 1:
        raise ValueError("Final MIST must declare a single-variable acquisition")
    for record in records:
        if record.encoded_record.mode != "json":
            raise ValueError(
                "MIST export requires structured committed point summaries"
            )
        point = _MIST_POINT.validate_python(record.encoded_record.value)
        if point.index < 0 or point.index >= len(fluxes) or point.index in seen:
            raise ValueError("MIST committed index must be unique and inside the plan")
        seen.add(point.index)
        fluxes[point.index] = point.flux
        trace = _load_declared_trace(
            data_root.resolve(), record, point.gain_file, spec.tag, schemas[0]
        )
        if trace is None:
            continue
        if not np.array_equal(trace.axes[0].values, gains):
            raise ValueError("MIST gain grid differs from the fixed plan")
        usable = True
        signal[point.index] = trace.z
        if point.success:
            success[point.index] = 1.0
    if not usable:
        raise ValueError("No usable committed MIST acquisition")
    flux_axis = Axis("Flux", plan.flux_unit, fluxes)
    return PreparedFluxdepExport(
        ExperimentPayload(
            variables={
                DataVariable("signal"): LabberPayload(
                    ("Signal", "a.u.", signal),
                    [Axis("Readout Gain", "a.u.", gains), flux_axis],
                ),
                DataVariable("success"): LabberPayload(
                    ("Success", "", success), [flux_axis]
                ),
            },
            metadata=LabberMetadata(
                comment="Committed MIST raw gain traces and acquisition success"
            ),
            representation="grouped",
        ),
        CfgSnapshot(
            values=plan.model_dump(mode="json"),
            cfg_type="MISTPlan",
            schema_version="1.0",
        ),
    )
