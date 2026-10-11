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
    load_run_data,
    validate_experiment_payload,
)
from zcu_tools.experiment.axes_spec import AxesSpec
from zcu_tools.experiment.native_persistence import native_schemas
from zcu_tools.experiment.workflows import CommittedRecord
from zcu_tools.experiment.workflows.exporting import PreparedFluxdepExport

from zcu_lab.v2.twotone.time_domain.t1.core import T1Exp

from .t1 import T1Plan, T1Point

_POINT = TypeAdapter(T1Point)


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
    paths, invalid native T1 data, or multiple final traces per point raise
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
        trace = _load_final_t1(root, record)
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


def _load_final_t1(root: Path, record: CommittedRecord) -> LabberPayload | None:
    """Select only the declared final decay from this committed invocation."""
    spec = T1Exp().persistence_spec()
    if not isinstance(spec, AxesSpec):
        raise ValueError("Final T1 must declare a single-variable acquisition")
    final: LabberPayload | None = None
    for relative in record.run_files:
        path = (root / record.iteration_dir / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("committed acquisition path escapes data_root")
        stored = load_run_data(path)
        if stored.metadata.experiment != spec.tag:
            continue
        validate_experiment_payload(stored.payload, schema=native_schemas(spec))
        trace = stored.payload.variables[spec.data_variable]
        if len(trace.z) == 0:
            continue
        if final is not None:
            raise ValueError("multiple final T1 acquisitions for one committed point")
        final = trace
    return final
