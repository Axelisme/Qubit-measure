"""Independent MIST run conversion to Labber, not samples."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from zcu_tools.datafile import (
    Axis,
    Container,
    ContainerAttrs,
    LabberMetadata,
    Variable,
)
from zcu_tools.experiment.metadata import CfgSnapshot
from zcu_tools.experiment.native_persistence import PreparedRunData
from zcu_tools.experiment.workflows.export import (
    FluxdepExportOutput,
    load_trace,
    write_outputs,
)
from zcu_tools.experiment.workflows.saved_run import (
    SavedRecord,
    read_manifest,
    read_records,
)

from zcu_lab.v2.mist.power_dep.single_trace.core import PowerDepExp
from zcu_lab.workflows.mist_fluxdep.workflow import MISTPlan, MISTPoint


@dataclass(frozen=True)
class MISTExportReport:
    """Report raw MIST export results so callers can identify missing science inputs.

    outputs lists the selected Labber writer result, including path,
    status and failure reason. missing_records remains an empty tuple because
    saved_run only returns committed records, without reconstructing missing ones.
    """

    outputs: tuple[FluxdepExportOutput, ...]
    missing_records: tuple[Path, ...]


def prepare_export(
    plan: MISTPlan,
    points: tuple[SavedRecord[MISTPoint], ...],
    *,
    mode: Literal["excited", "ground"] | None = None,
) -> PreparedRunData:
    """Assemble raw MIST response across flux and gain for offline analysis.

    Stopped partial traces retain their native reference with success=False.
    They export raw IQ, including NaN, without contributing an acquisition-success
    value. Complete and stopped native acquisitions are accepted; other completion
    states reject.

    Return Flux×actualgain Signal and Flux-only Success (1 or NaN).
    mode selects a preparation; None selects ground for a ground-only plan and
    excited otherwise. Historical single-trace records retain their native file.

    plan is the fixed manifest plan; points are structured saved commits. Only
    named final mist/gain files enter the payload. Missing rows retain NaN.
    Actual gain must equal the plan grid exactly, without rounding/union or
    interpolation. No usable trace, bad index/reference or schema mismatch raises
    before outputs are allocated; the source record/native path is diagnosed.
    """
    mode = (
        ("ground" if plan.measure_mode == "ground" else "excited")
        if mode is None
        else mode
    )
    if plan.measure_mode != "both" and mode != plan.measure_mode:
        raise ValueError(f"Preparation {mode!r} was not requested by the MIST plan")
    gains = np.linspace(plan.gain_start, plan.gain_stop, plan.gain_points)
    fluxes = np.array(plan.fluxes)
    signal = np.full((len(fluxes), len(gains)), complex(np.nan, np.nan))
    success = np.full(len(fluxes), np.nan)
    seen: set[int] = set()
    usable = False
    for saved in points:
        point = saved.record
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
            selected = point.traces.get(mode)
            gain_file = selected.gain_file if selected is not None else point.gain_file
            completed = selected.success if selected is not None else point.success
            if point.traces and selected is None:
                continue
            if completed and gain_file is None:
                raise ValueError("MIST success must identify its Completed acquisition")
            fluxes[point.index] = point.flux
            if gain_file is None:
                continue
            trace = load_trace(
                saved.directory,
                gain_file,
                PowerDepExp.AXES_SPEC,
                allow_stopped=True,
            )
            if not np.array_equal(trace.axes[0].values, gains):
                raise ValueError("MIST gain grid differs from the fixed plan")
            usable = True
            signal[point.index] = trace.z
            if completed:
                success[point.index] = 1.0
        except ValueError as error:
            raise ValueError(f"{saved.record_path}: {error}") from error
    if not usable:
        raise ValueError("No usable committed MIST acquisition")
    flux_axis = Axis("Flux", plan.flux_unit, fluxes)
    return PreparedRunData(
        Container(
            vars={
                "signal": Variable(
                    signal, (flux_axis, Axis("Readout Gain", "a.u.", gains)), "a.u."
                ),
                "success": Variable(success, (flux_axis,), ""),
            },
            docs={},
            groups={},
            attrs=ContainerAttrs("3.0", "zcu_lab", "workflow-export", {}),
        ),
        CfgSnapshot(
            values=plan.model_dump(mode="json"),
            cfg_type="MISTPlan",
            schema_version="1.0",
        ),
        LabberMetadata(
            comment=f"Committed MIST {mode} raw gain traces and acquisition success"
        ),
        "independent",
    )


def export_run(
    run: Path,
    *,
    database: Path | None = None,
    mode: Literal["excited", "ground"] | None = None,
) -> MISTExportReport:
    """Export raw MIST response across flux and gain for offline analysis.

    run is the metadata directory. Providing database writes Labber; omitting
    it raises before any write. Samples is not supported. mode selects
    excited/ground; None selects ground for a ground-only plan and excited
    otherwise. Exporting both modes requires separate calls, each writing a new
    Labber file.
    Fixed manifest and each record.json provide input, not journal or leftover
    files. Missing records are not reconstructed or listed; malformed inputs
    name their source.
    Prepare errors allocate nothing; writers return independent typed failures.
    """
    if database is None:
        raise ValueError("Provide --database for Labber")
    manifest = read_manifest(run, "mist_fluxdep")
    try:
        plan = MISTPlan.model_validate(manifest.plan)
        if str(plan.snapshot.entry_id) != manifest.identity.entry_id:
            raise ValueError("Plan snapshot entry and manifest identity differ")
    except ValueError as error:
        raise ValueError(f"{run}/manifest.json: {error}") from error
    points = read_records(run, MISTPoint)
    prepared = prepare_export(plan, points, mode=mode)
    outputs_written = write_outputs(
        manifest,
        prepared,
        plan.snapshot,
        database=database,
    )
    return MISTExportReport(outputs_written, ())


def main(argv: list[str] | None = None) -> int:
    """Export saved MIST flux/gain data for offline analysis from CLI arguments.

    Return 1 on an isolated writer failure and 0 when all writers succeed.

    argv omits the executable. --database writes Labber and is required.
    Malformed inputs raise; writer results are printed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--mode", choices=("excited", "ground"))
    args = parser.parse_args(argv)
    if args.database is None:
        parser.error("Provide --database for Labber")
    report = export_run(args.run, database=args.database, mode=args.mode)
    for source in report.missing_records:
        print(f"Missing record: {source}")
    for out in report.outputs:
        print(f"{out.format}: {out.status}: {out.path}: {out.error or ''}")
    return int(any(out.status == "failed" for out in report.outputs))


if __name__ == "__main__":
    raise SystemExit(main())
