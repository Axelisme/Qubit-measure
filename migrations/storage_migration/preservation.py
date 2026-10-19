"""Route byte-identical legacy files and preflight their owned publication."""

from dataclasses import dataclass, replace
from pathlib import Path

from migrations.sample_csv import plan_sample_csv

from .errors import MigrationInputError
from .models import (
    MigrationDeclarations,
    MigrationDestination,
    MigrationManifest,
    MigrationSource,
)
from .paths import contained_path, source_removed, validate_segment
from .state import MigrationSession, clear_pending, file_hash, record_pending

_RESULT_RESERVED = frozenset(
    {
        "setup.yaml",
        "module_cfg.yaml",
        "aliases.yaml",
        "points",
        "records",
        "figures",
        "samples",
        "predictors",
        "dispersive",
        "t1_curve",
        "runs",
    }
)
_POINT_RESERVED = frozenset({"point.yaml", "module_cfg.yaml", "figures"})
_DATABASE_RESERVED = frozenset({"runs", "Labber"})


@dataclass(frozen=True)
class _CopyTarget:
    """One raw source/destination, optional required point and reserved conflict."""

    source: Path
    destination: Path
    point: str | None = None
    reserved: bool = False


def _result_target(
    path: Path,
    source: MigrationSource,
    destination: MigrationDestination,
    declarations: MigrationDeclarations,
) -> _CopyTarget | None:
    relative = path.relative_to(source.result_path)
    parts = relative.parts
    top = parts[0]
    if relative == Path("params.json"):
        return None
    if relative == Path("samples.csv"):
        plan = plan_sample_csv(
            [path],
            dev_value_column=declarations.sample_value_column,
            source_unit=declarations.sample_source_unit,
        )
        return (
            _CopyTarget(path, destination.result_path / relative)
            if plan.pending
            else None
        )
    if path.name in ("meta_info.json", "module_cfg.yaml") and (
        len(parts) == 1
        or (len(parts) == 2 and (path.parent / "meta_info.json").is_file())
        or (len(parts) == 3 and parts[0] == "exps")
    ):
        return None
    if len(parts) >= 3 and parts[0] == "exps":
        label = parts[1]
        validate_segment(label, "point label")
        point_relative = Path(*parts[2:])
        figure = (
            len(parts) == 4 and parts[2] == "image" and path.suffix.lower() == ".png"
        )
        target = Path("figures") / path.name if figure else point_relative
        return _CopyTarget(
            path,
            destination.result_path / "points" / label / target,
            label,
            not figure and point_relative.parts[0] in _POINT_RESERVED,
        )
    if (
        "image" in parts[:-1]
        and "exps" not in parts[:-1]
        and "autofluxdep_runs" not in parts[:-1]
    ):
        return _CopyTarget(path, destination.result_path / "figures" / relative)
    return _CopyTarget(
        path, destination.result_path / relative, reserved=top in _RESULT_RESERVED
    )


def _copy_targets(
    source: MigrationSource,
    destination: MigrationDestination,
    declarations: MigrationDeclarations,
) -> list[_CopyTarget]:
    targets = []
    for path in sorted(source.result_path.rglob("*")):
        if path.is_file():
            path = contained_path(path, source.result_path)
            target = _result_target(path, source, destination, declarations)
            if target is not None:
                targets.append(target)
    for path in sorted(source.database_path.rglob("*")):
        if not path.is_file():
            continue
        path = contained_path(path, source.database_path)
        relative = path.relative_to(source.database_path)
        labber = path.suffix.lower() in (".h5", ".hdf5")
        target = Path("Labber") / relative if labber else relative
        targets.append(
            _CopyTarget(
                path,
                destination.database_path / target,
                reserved=not labber and relative.parts[0] in _DATABASE_RESERVED,
            )
        )
    return targets


def _check_existing(
    target: _CopyTarget,
    destination: MigrationDestination,
    manifest: MigrationManifest | None,
) -> None:
    path = target.destination
    for ancestor in path.parents:
        if ancestor.is_symlink() or (ancestor.exists() and not ancestor.is_dir()):
            raise FileExistsError(
                f"{target.source} -> {path}: occupied ancestor {ancestor}"
            )
    if target.point is not None:
        point = destination.result_path / "points" / target.point
        if point.exists() and not (
            (point / "point.yaml").is_file() and (point / "module_cfg.yaml").is_file()
        ):
            raise FileExistsError(
                f"{target.source} -> {path}: occupied incomplete point directory {point}"
            )
    if not (path.exists() or path.is_symlink()):
        return
    owned = next(
        (
            state
            for state in (() if manifest is None else manifest.files)
            if state.source == target.source
            and state.destination == path
            and state.operation in ("copy", "move_labber")
            and state.phase in ("prepared", "published", "source_removed")
        ),
        None,
    )
    if owned is None or path.is_symlink() or not path.is_file():
        raise FileExistsError(
            f"{target.source} -> {path}: unowned destination collision"
        )
    if owned.destination_hash != file_hash(path):
        raise MigrationInputError(
            f"{target.source} -> {path}: destination hash changed"
        )


def preflight_preservation(
    source: MigrationSource,
    destination: MigrationDestination,
    declarations: MigrationDeclarations,
    *,
    manifest: MigrationManifest | None = None,
) -> None:
    """Refuse conflicting raw-copy routes before backup, entry or report writes.

    source/destination are resolved entry roots. declarations go unchanged to the
    CSV owner's pure plan. Reject reserved native files/directories, overlapping
    source routes and occupied destinations/ancestors with both source and target
    locations. Only matching manifest-owned, hash-verified publications can reuse
    existing destinations, including recovery of an already prepared publication.
    No writes occur; invalid paths, owner planning and I/O errors propagate.
    """
    targets = _copy_targets(source, destination, declarations)
    claimed: dict[Path, _CopyTarget] = {}
    for target in sorted(targets, key=lambda item: item.destination):
        previous = claimed.get(target.destination)
        if previous is None:
            previous = next(
                (
                    claimed[parent]
                    for parent in target.destination.parents
                    if parent in claimed
                ),
                None,
            )
        if previous is not None:
            raise MigrationInputError(
                f"{previous.source} -> {previous.destination} conflicts with "
                f"{target.source} -> {target.destination}"
            )
        claimed[target.destination] = target
    for target in targets:
        if target.reserved:
            raise MigrationInputError(
                f"{target.source} -> {target.destination}: reserved native destination"
            )
        _check_existing(target, destination, manifest)


def _publish_target(session: MigrationSession, target: _CopyTarget) -> None:
    if target.point is not None:
        root = session.manifest.report.destination.result_path
        point = root / "points" / target.point
        ready = (point / "point.yaml").is_file() and (
            point / "module_cfg.yaml"
        ).is_file()
        if not ready and not (session.dry_run and session.manifest.parameters_complete):
            record_pending(
                session,
                target.source,
                "point files",
                f"Point {target.point!r} is not created; run parameters then resume data",
            )
            return
    state = session.publish(target.source, target.destination, operation="copy")
    session.completed(state)
    clear_pending(session, target.source, "point files")


def preserve_result_files(session: MigrationSession) -> None:
    """Publish result raw copies at entry/point-relative paths, keeping bytes/hash.

    session owns the verified sources, manifest and fixed declarations. Supported
    parameter files stay with their converter; partial CSVs also retain their raw
    source through the CSV owner's pending plan. Existing figure routing remains.
    Missing points stay located pending without creating their directory. Dry-run
    only plans. Hash/collision/I/O errors propagate with owned recovery state.
    """
    report = session.manifest.report
    for target in _copy_targets(report.source, report.destination, report.declarations):
        if target.source.is_relative_to(report.source.result_path):
            _publish_target(session, target)


def preserve_database_files(session: MigrationSession) -> None:
    """Copy Database sources not already moved by validated native conversion.

    All .h5/.hdf5 originals use Labber/<source-relative-path> regardless of native
    success. Other unknown files keep entry-relative paths. Raw copying records
    preservation, never conversion, and leaves run-evidence pending unchanged.
    Dry-run plans only; publication conflicts and I/O failures propagate.
    """
    report = session.manifest.report
    for target in _copy_targets(report.source, report.destination, report.declarations):
        if target.source.is_relative_to(
            report.source.database_path
        ) and not source_removed(target.source, session.manifest):
            _publish_target(session, target)


def report_unrecognized_files(session: MigrationSession) -> None:
    """Refresh pending locations for source files untouched by any owned action.

    Remove obsolete unconverted-file locations, keep explicit converter pending,
    and checkpoint the cumulative report. Sources are unchanged; I/O propagates.
    """
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
                        "No authorized conversion; original bytes remain in verified backup",
                    )
