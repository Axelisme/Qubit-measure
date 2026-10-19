"""Offline migration preservation ownership."""

from dataclasses import replace
from pathlib import Path

from .paths import contained_path, source_removed, validate_segment
from .state import (
    MigrationSession,
    clear_pending,
    record_pending,
)


def _publish_point_figure(session: MigrationSession, path: Path) -> bool:
    source = session.manifest.report.source.result_path
    relative = path.relative_to(source)
    parts = relative.parts
    if (
        len(parts) != 4
        or parts[0] != "exps"
        or parts[2] != "image"
        or path.suffix.lower() != ".png"
    ):
        return False
    label = parts[1]
    validate_segment(label, "point label")
    destination = session.manifest.report.destination.result_path
    point = contained_path(destination / "points" / label, destination)
    point_exists = (point / "point.yaml").is_file() and (
        point / "module_cfg.yaml"
    ).is_file()
    if not point_exists and not (
        session.dry_run and session.manifest.parameters_complete
    ):
        record_pending(
            session,
            contained_path(path, source),
            "point figures",
            f"Point {label!r} is not created; run parameters then resume data",
        )
        return True
    state = session.publish(
        contained_path(path, source),
        contained_path(point / "figures" / path.name, destination),
        operation="copy",
    )
    session.completed(state)
    clear_pending(session, path.resolve(), "point figures")
    return True


def preserve_result_files(session: MigrationSession) -> None:
    """Copy authorized waveform/sample/image/workflow source bytes through owned publication.

    session identifies the old result and new Database entry. Preserve relative
    paths and hashes, leave originals intact and report unproven associations.
    exps/<point>/image/*.png publishes to that point's result figures instead of
    an archive. Missing points stay pending until parameters then data resume.
    dry_run plans only; collision/hash/I/O failures propagate with recovery state."""
    source = session.manifest.report.source.result_path
    destination = session.manifest.report.destination.database_path
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        if _publish_point_figure(session, path):
            continue
        relative = path.relative_to(source)
        archive = next(
            (
                part
                for part in relative.parts[:-1]
                if part in ("arb_waveforms", "autofluxdep_runs", "image")
            ),
            None,
        )
        if archive is None and relative != Path("samples.csv"):
            continue
        target = (
            destination / relative
            if archive == "arb_waveforms"
            else destination / "migration-preserved" / "result" / relative
        )
        state = session.publish(
            contained_path(path, source),
            contained_path(target, destination),
            operation="copy",
        )
        session.completed(state)
        if archive != "arb_waveforms":
            reason = (
                "Archive preserved; point/run association is unproven"
                if archive is not None
                else (
                    "No proven component/sample/point association; not a runtime v2 SampleTable"
                )
            )
            record_pending(session, path.resolve(), "association", reason)


def report_unrecognized_files(session: MigrationSession) -> None:
    """Refresh pending locations for untouched source files in the session's report.

    Previously unrecognized files that are now identified cease to be pending.
    No source is changed. Checkpoint the cumulative report; I/O failures propagate."""
    source = session.manifest.report.source
    report = session.manifest.report
    session.report(
        replace(
            report,
            pending=tuple(
                item for item in report.pending if item.location != "unconverted file"
            ),
        )
    )
    known = set(session.manifest.source_hashes)
    known.update(str(item.source) for item in session.manifest.report.pending)
    for root in (source.result_path, source.database_path):
        for candidate in sorted(root.rglob("*")):
            if source_removed(candidate, session.manifest):
                continue
            if candidate.is_file():
                path = contained_path(candidate, root)
                if str(path) not in known:
                    record_pending(
                        session,
                        path,
                        "unconverted file",
                        "File remains at its original source; no authorized conversion",
                    )
