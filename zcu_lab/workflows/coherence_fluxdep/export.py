"""Independent Coherence run conversion: samples, merged native and Labber."""

import argparse
from dataclasses import dataclass
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
)
from zcu_tools.experiment.workflows.export import (
    FluxdepExportOutput,
    PreparedFluxdepExport,
    validate_destinations,
    write_outputs,
)
from zcu_tools.experiment.workflows.saved_run import (
    SavedManifest,
    SavedPoint,
    read_manifest,
    read_points,
)
from zcu_tools.resources.entry.storage import bind_sample_table

from zcu_lab.workflows.coherence_fluxdep.lifetime_export import load_lifetime_trace
from zcu_lab.workflows.coherence_fluxdep.measurements import (
    LifetimeKind,
    LifetimeResult,
)
from zcu_lab.workflows.coherence_fluxdep.samples import (
    SampleExportReport,
    append_samples,
    prepare_samples,
)
from zcu_lab.workflows.coherence_fluxdep.workflow import CoherencePlan, CoherencePoint


@dataclass(frozen=True)
class CoherenceExportReport:
    """This call's writer receipts, optional samples receipt and all absent records.

    missing_records lists exact record.json paths; those invocations cannot be
    recovered. outputs retains independent selected native/Labber failures.
    Samples I/O failure raises SampleExportError with successful prefix instead.
    """

    outputs: tuple[FluxdepExportOutput, ...]
    samples: SampleExportReport | None
    missing_records: tuple[Path, ...]


def _validate_lifetime_evidence(item: LifetimeResult) -> None:
    """Reject inconsistent completion files or accepted lifetime/error scalars."""
    if (item.status in {"accepted", "rejected"}) != (item.file is not None):
        raise ValueError(
            "Completed lifetime requires file; failed/skipped cannot name one"
        )
    if item.status == "accepted":
        if (
            item.lifetime_us is None
            or not np.isfinite(item.lifetime_us)
            or item.lifetime_us <= 0
            or item.error_us is None
            or not np.isfinite(item.error_us)
            or item.error_us < 0
        ):
            raise ValueError(
                "Accepted lifetime/error must be positive/nonnegative finite"
            )
    elif item.lifetime_us is not None or item.error_us is not None:
        raise ValueError("Nonaccepted lifetime cannot carry accepted scalars")


def _points(
    manifest: SavedManifest, plan: CoherencePlan
) -> tuple[tuple[SavedPoint[CoherencePoint], ...], tuple[Path, ...]]:
    points, missing = read_points(manifest, TypeAdapter(CoherencePoint))
    seen: set[int] = set()
    expected = tuple(
        kind
        for kind, enabled in (
            ("t1", plan.measure_t1),
            ("t2ramsey", plan.measure_t2ramsey),
            ("t2echo", plan.measure_t2echo),
        )
        if enabled
    )
    for saved in points:
        point = saved.point
        try:
            if (
                point.index < 0
                or point.index >= len(plan.fluxes)
                or point.index in seen
                or not np.isfinite(point.flux)
            ):
                raise ValueError(
                    "Point index must be unique/inside plan and flux finite"
                )
            seen.add(point.index)
            if tuple(item.kind for item in point.measurements) != expected:
                raise ValueError(
                    "Point lifetime order/selection differs from fixed plan"
                )
            for item in point.measurements:
                _validate_lifetime_evidence(item)
        except ValueError as error:
            raise ValueError(f"{saved.record_path}: {error}") from error
    return points, missing


def _channels(
    plan: CoherencePlan,
    points: tuple[SavedPoint[CoherencePoint], ...],
    kind: LifetimeKind,
) -> dict[DataVariable, LabberPayload]:

    fluxes = np.array(plan.fluxes)
    lifetimes = np.full(len(fluxes), np.nan)
    traces: dict[int, LabberPayload] = {}
    for saved in points:
        point = saved.point
        fluxes[point.index] = point.flux
        item = next(item for item in point.measurements if item.kind == kind)
        if item.file is None:
            continue
        traces[point.index] = load_lifetime_trace(saved.directory, item.file, item.kind)
        if item.status == "accepted" and item.lifetime_us is not None:
            lifetimes[point.index] = item.lifetime_us / 1e6
    if not traces:
        return {}
    count = max(len(trace.z) for trace in traces.values())
    signal = np.full((len(fluxes), count), complex(np.nan, np.nan))
    time = np.full(signal.shape, np.nan)
    for index, trace in traces.items():
        signal[index, : len(trace.z)] = trace.z
        time[index, : len(trace.z)] = trace.axes[0].values
    flux_axis = Axis("Flux", plan.flux_unit, fluxes)
    axes = [Axis(kind + " Time Index", "", np.arange(count)), flux_axis]
    return {
        DataVariable(kind + "_signal"): LabberPayload(
            (kind + " Signal", "a.u.", signal), axes
        ),
        DataVariable(kind + "_time"): LabberPayload((kind + " Time", "s", time), axes),
        DataVariable(kind): LabberPayload((kind, "s", lifetimes), [flux_axis]),
    }


def prepare_export(
    plan: CoherencePlan, points: tuple[SavedPoint[CoherencePoint], ...]
) -> PreparedFluxdepExport:
    """Prepare each enabled, actually measured lifetime on its own index grid.

    points are typed, validated saved commits, never Engine.records or journal.
    Each measured kind has Signal/Time Flux×index and Flux-only lifetime seconds;
    rejected fits keep raw and NaN lifetime. Disabled/unmeasured kinds have no
    channels. Missing rows/short traces are NaN; no union/round/interpolation or
    calibration traces enter this payload. No usable trace raises before writes.
    """
    variables: dict[DataVariable, LabberPayload] = {}
    selection: tuple[tuple[LifetimeKind, bool], ...] = (
        ("t1", plan.measure_t1),
        ("t2ramsey", plan.measure_t2ramsey),
        ("t2echo", plan.measure_t2echo),
    )
    for kind, enabled in selection:
        if enabled:
            variables.update(_channels(plan, points, kind))
    if not variables:
        raise ValueError("No usable committed Coherence acquisition")
    return PreparedFluxdepExport(
        ExperimentPayload(
            variables=variables,
            metadata=LabberMetadata(
                comment="Committed Coherence raw traces and accepted lifetimes"
            ),
            representation="grouped",
        ),
        CfgSnapshot(
            values=plan.model_dump(mode="json"),
            cfg_type="CoherencePlan",
            schema_version="1.0",
        ),
    )


def export_run(
    run: Path,
    *,
    outputs: tuple[str, ...] = ("merged",),
    result: Path | None = None,
    database: Path | None = None,
) -> CoherenceExportReport:
    """Convert one saved coherence_fluxdep run without a GUI, engine or hardware.

    run is its exact metadata directory. outputs selects unique samples/merged/
    labber. Samples requires explicit entry result; Labber explicit Database;
    merged needs neither. Missing destinations reject before ANY write. Fixed
    manifest identity/plan/roots/times and structured iteration records provide
    input; missing records are listed in the receipt, never recovered. Malformed
    input/native data raises source-named errors before output allocation. Prepare
    ALL selected data/rows first. Writers fail independently; samples append
    retains prefix/count/cause and latest-wins through its existing owner.
    """
    validate_destinations(
        outputs, {"samples", "merged", "labber"}, result=result, database=database
    )
    manifest = read_manifest(run, "coherence_fluxdep")
    try:
        plan = CoherencePlan.model_validate(manifest.plan)
        if str(plan.snapshot.entry_id) != manifest.identity.entry_id:
            raise ValueError("Plan snapshot entry and manifest identity differ")
    except ValueError as error:
        raise ValueError(f"{run}/manifest.json: {error}") from error
    points, missing = _points(manifest, plan)
    rows = (
        prepare_samples(manifest.identity.run_id, plan, points)
        if "samples" in outputs
        else ()
    )
    prepared = (
        prepare_export(plan, points) if {"merged", "labber"} & set(outputs) else None
    )
    table = None
    if "samples" in outputs and result is not None:
        table = bind_sample_table(result, plan.snapshot.roles["qubit"])
    written = (
        ()
        if prepared is None
        else write_outputs(
            manifest,
            prepared,
            plan.snapshot,
            merged="merged" in outputs,
            database=database if "labber" in outputs else None,
        )
    )
    samples = (
        None
        if table is None
        else append_samples(
            manifest.identity.run_id, rows=rows, table=table, point=plan.snapshot.point
        )
    )
    return CoherenceExportReport(written, samples, missing)


def main(argv: list[str] | None = None) -> int:
    """CLI shell over export_run; print missing records and writer results.

    argv omits the executable name. Select --samples/--merged/--labber explicitly;
    no selection is a CLI error. --result/--database are required only when used.
    Return 1 on isolated writer failure, 0 otherwise; input/append errors raise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    for output in ("merged", "labber", "samples"):
        parser.add_argument("--" + output, action="store_true")
    parser.add_argument("--result", type=Path)
    parser.add_argument("--database", type=Path)
    args = parser.parse_args(argv)
    outputs = tuple(
        name for name in ("merged", "labber", "samples") if getattr(args, name)
    )
    if not outputs:
        parser.error("Select --merged, --labber or --samples")
    report = export_run(
        args.run, outputs=outputs, result=args.result, database=args.database
    )
    for source in report.missing_records:
        print(f"Missing record: {source}")
    for out in report.outputs:
        print(f"{out.format}: {out.status}: {out.path}: {out.error or ''}")
    if report.samples is not None:
        print(f"samples: {report.samples.written_count}: {report.samples.path}")
    return int(any(out.status == "failed" for out in report.outputs))


if __name__ == "__main__":
    raise SystemExit(main())
