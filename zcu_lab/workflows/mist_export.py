"""Independent MIST run conversion to merged native and Labber, not samples."""

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
    load_trace,
    validate_destinations,
    write_outputs,
)
from zcu_tools.experiment.workflows.saved_run import (
    SavedPoint,
    read_manifest,
    read_points,
)

from zcu_lab.v2.mist.power_dep.single_trace.core import PowerDepExp

from .mist import MISTPlan, MISTPoint


@dataclass(frozen=True)
class MISTExportReport:
    """This call's selected writer results and every absent iteration record path."""

    outputs: tuple[FluxdepExportOutput, ...]
    missing_records: tuple[Path, ...]


def prepare_export(
    plan: MISTPlan, points: tuple[SavedPoint[MISTPoint], ...]
) -> PreparedFluxdepExport:
    """Prepare raw Flux×actualgain Signal and Flux-only Success (1 or NaN).

    plan is the fixed manifest plan; points are structured saved commits. Only
    named final mist/gain files enter the payload. Missing rows retain NaN.
    Actual gain must equal the plan grid exactly, without rounding/union or
    interpolation. No usable trace, bad index/reference or schema mismatch raises
    before outputs are allocated; the source record/native path is diagnosed.
    """
    gains = np.linspace(plan.gain_start, plan.gain_stop, plan.gain_points)
    fluxes = np.array(plan.fluxes)
    signal = np.full((len(fluxes), len(gains)), complex(np.nan, np.nan))
    success = np.full(len(fluxes), np.nan)
    seen: set[int] = set()
    usable = False
    for saved in points:
        point = saved.point
        try:
            if (
                point.index < 0
                or point.index >= len(fluxes)
                or point.index in seen
                or not np.isfinite(point.flux)
            ):
                raise ValueError(
                    "MIST index must be unique/inside plan and flux finite"
                )
            seen.add(point.index)
            if point.success != (point.gain_file is not None):
                raise ValueError("MIST success must identify its Completed acquisition")
            fluxes[point.index] = point.flux
            if point.gain_file is None:
                continue
            trace = load_trace(
                saved.directory, point.gain_file, PowerDepExp().persistence_spec()
            )
            if not np.array_equal(trace.axes[0].values, gains):
                raise ValueError("MIST gain grid differs from the fixed plan")
            usable = True
            signal[point.index] = trace.z
            if point.success:
                success[point.index] = 1.0
        except ValueError as error:
            raise ValueError(f"{saved.record_path}: {error}") from error
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


def export_run(
    run: Path, *, outputs: tuple[str, ...] = ("merged",), database: Path | None = None
) -> MISTExportReport:
    """Convert one saved mist_fluxdep run, without GUI/engine/live devices.

    run is the metadata directory. Select merged and/or labber; samples and all
    other selections reject. Labber requires explicit --database before any
    write. Merged replaces data_root/merged.h5 with no external destination.
    Fixed manifest and each record.json provide input, not journal or leftover
    files. Missing records are listed; malformed inputs name their source.
    Prepare errors allocate nothing; writers return independent typed failures.
    """
    validate_destinations(outputs, {"merged", "labber"}, result=None, database=database)
    manifest = read_manifest(run, "mist_fluxdep")
    try:
        plan = MISTPlan.model_validate(manifest.plan)
        if str(plan.snapshot.entry_id) != manifest.identity.entry_id:
            raise ValueError("Plan snapshot entry and manifest identity differ")
    except ValueError as error:
        raise ValueError(f"{run}/manifest.json: {error}") from error
    points, missing = read_points(manifest, TypeAdapter(MISTPoint))
    prepared = prepare_export(plan, points)
    outputs_written = write_outputs(
        manifest,
        prepared,
        plan.snapshot,
        merged="merged" in outputs,
        database=database if "labber" in outputs else None,
    )
    return MISTExportReport(outputs_written, missing)


def main(argv: list[str] | None = None) -> int:
    """CLI shell over export_run, returning 1 on an isolated writer failure.

    argv omits the executable. Select --merged and/or --labber. --database is
    necessary only for Labber. Unsupported --samples is an argparse error.
    Malformed inputs raise; every missing record and writer result is printed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--merged", action="store_true")
    parser.add_argument("--labber", action="store_true")
    parser.add_argument("--database", type=Path)
    args = parser.parse_args(argv)
    outputs = tuple(name for name in ("merged", "labber") if getattr(args, name))
    if not outputs:
        parser.error("Select --merged and/or --labber")
    report = export_run(args.run, outputs=outputs, database=args.database)
    for source in report.missing_records:
        print(f"Missing record: {source}")
    for out in report.outputs:
        print(f"{out.format}: {out.status}: {out.path}: {out.error or ''}")
    return int(any(out.status == "failed" for out in report.outputs))


if __name__ == "__main__":
    raise SystemExit(main())
