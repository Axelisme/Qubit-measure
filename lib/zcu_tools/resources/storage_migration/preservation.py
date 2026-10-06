"""Offline migration preservation ownership."""

from dataclasses import replace
from pathlib import Path

from .paths import contained_path
from .state import (
    MigrationSession,
    record_pending,
)


def preserve_result_files(session: MigrationSession) -> None:
    """Copy authorized waveform/sample/image/workflow source bytes through owned publication.

    session identifies the old result and new Database entry. Preserve relative
    paths and hashes, leave originals intact and report unproven associations.
    dry_run plans only; collision/hash/I/O failures propagate with recovery state."""
    source = session.manifest.report.source.result_path
    destination = session.manifest.report.destination.database_path
    for path in sorted(source.rglob("*")):
        if not path.is_file():
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
            if candidate.is_file():
                path = contained_path(candidate, root)
                if str(path) not in known:
                    record_pending(
                        session,
                        path,
                        "unconverted file",
                        "File remains at its original source; no authorized conversion",
                    )
