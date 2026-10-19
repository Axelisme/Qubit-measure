"""Complete external source backups and hash-guarded, resumable source cleanup."""

import shutil
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from .errors import MigrationInputError
from .models import MigrationBackup, MigrationRequest, MigrationSource, PendingItem
from .state import MigrationSession, file_hash


@dataclass(frozen=True, kw_only=True)
class BackupInventory:
    """Complete backup listing relative to its directory, without symlinks.

    hashes maps result/<relative> and Database/<relative> files to lowercase
    SHA256. directories records all directories, including empty source roots.
    """

    hashes: Mapping[str, str]
    directories: tuple[str, ...]


_INVENTORY = TypeAdapter(BackupInventory)


def backup_location(request: MigrationRequest, source: MigrationSource) -> Path:
    """Resolve an external <name>-backup path without writing or allowing overlaps.

    request.repo_root defaults to the tool checkout. Reject a backup within that
    repo, enclosing either source, or overlapping either destination or report.
    Existence is checked separately so resume can verify the owned backup.
    """
    repo = (request.repo_root or Path(__file__).resolve().parents[2]).resolve()
    path = ((request.backup_parent or repo.parent) / f"{request.name}-backup").resolve()
    targets = (
        source.result_path,
        source.database_path,
        request.results_root.resolve() / request.name,
        request.database_root.resolve() / request.name,
    )
    if path.is_relative_to(repo) or any(
        path.is_relative_to(root) or root.is_relative_to(path) for root in targets
    ):
        raise MigrationInputError(
            f"{path}: backup must be outside repo and source/destination trees"
        )
    if request.report_path is not None and request.report_path.resolve().is_relative_to(
        path
    ):
        raise MigrationInputError(f"{path}: backup cannot contain the migration report")
    return path


def _inventory(
    roots: Mapping[str, Path], *, missing_ok: bool = False
) -> BackupInventory:
    hashes: dict[str, str] = {}
    directories: list[str] = []
    for label, root in roots.items():
        if root.is_symlink():
            raise MigrationInputError(f"{root}: backup/cleanup refuses symlinks")
        if not root.exists() and missing_ok:
            continue
        if not root.is_dir():
            raise MigrationInputError(f"{root}: backup source directory missing")
        directories.append(label)
        for path in sorted(root.rglob("*")):
            relative = f"{label}/{path.relative_to(root).as_posix()}"
            if path.is_symlink():
                raise MigrationInputError(f"{path}: backup/cleanup refuses symlinks")
            if path.is_dir():
                directories.append(relative)
            elif path.is_file():
                hashes[relative] = file_hash(path)
            else:
                raise MigrationInputError(
                    f"{path}: backup/cleanup requires regular files"
                )
    return BackupInventory(hashes=hashes, directories=tuple(sorted(directories)))


def _source_roots(source: MigrationSource) -> dict[str, Path]:
    return {"result": source.result_path, "Database": source.database_path}


def create_backup(
    path: Path, source: MigrationSource, *, dry_run: bool
) -> MigrationBackup | None:
    """Copy both complete source trees and publish sha256.json after byte verification.

    Refuse existing paths, symlinks or non-regular source files. dry_run validates
    the inventory but writes nothing and returns None. Failures leave any partial
    backup for explicit operator review, never grant migration permission.
    """
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    inventory = _inventory(_source_roots(source))
    if dry_run:
        return None
    path.mkdir(parents=True, exist_ok=False)
    for label, root in _source_roots(source).items():
        shutil.copytree(root, path / label, copy_function=shutil.copy2)
    copied = _inventory({label: path / label for label in ("result", "Database")})
    if copied != inventory or _inventory(_source_roots(source)) != inventory:
        raise MigrationInputError(
            f"{path}: backup verification failed or sources changed"
        )
    inventory_path = path / "sha256.json"
    with inventory_path.open("xb") as stream:
        stream.write(_INVENTORY.dump_json(inventory, indent=2))
    backup = MigrationBackup(
        path=path,
        inventory_hash=file_hash(inventory_path),
        source_hashes=_absolute_hashes(inventory, source),
    )
    verify_backup(backup)
    return backup


def verify_backup(backup: MigrationBackup) -> BackupInventory:
    """Verify the bound inventory and every backup file, independent of old sources.

    Return the complete typed listing. Reject changed lists, hashes, extra files,
    missing directories and symlinks with MigrationInputError; I/O propagates.
    No backup is rewritten, even when migration has already moved source files.
    """
    path = backup.path
    inventory_path = path / "sha256.json"
    if path.is_symlink() or inventory_path.is_symlink():
        raise MigrationInputError(f"{path}: backup symlink conflict")
    if file_hash(inventory_path) != backup.inventory_hash:
        raise MigrationInputError(f"{path}: backup inventory hash changed")
    try:
        inventory = _INVENTORY.validate_json(inventory_path.read_bytes(), strict=True)
    except ValidationError as exc:
        raise MigrationInputError(f"{path}: invalid backup inventory") from exc
    actual = _inventory({label: path / label for label in ("result", "Database")})
    if actual != inventory or {item.name for item in path.iterdir()} != {
        "result",
        "Database",
        "sha256.json",
    }:
        raise MigrationInputError(f"{path}: backup contents differ from inventory")
    return inventory


def _absolute_hashes(
    inventory: BackupInventory, source: MigrationSource
) -> dict[str, str]:
    roots = _source_roots(source)
    hashes: dict[str, str] = {}
    for relative, digest in inventory.hashes.items():
        label, name = relative.split("/", 1)
        hashes[str(roots[label] / name)] = digest
    return hashes


def validate_backup_sources(session: MigrationSession) -> BackupInventory:
    """Verify backup and reject new/changed/unexpectedly missing source data.

    Before cleanup, only published Labber moves may be absent. After the cleaning
    checkpoint, missing backed-up files/directories are resumable. New files,
    directories, symlinks or changed bytes always block cleanup, never get deleted.
    """
    backup = session.manifest.backup
    if backup is None:
        raise MigrationInputError("manifest: verified backup is required")
    inventory = verify_backup(backup)
    source = session.manifest.report.source
    if backup.source_hashes != _absolute_hashes(inventory, source):
        raise MigrationInputError(
            "manifest: backup source hashes disagree with inventory"
        )
    current = _inventory(
        _source_roots(source), missing_ok=session.manifest.cleanup_phase != "pending"
    )
    for relative, digest in current.hashes.items():
        if inventory.hashes.get(relative) != digest:
            raise MigrationInputError(
                f"{relative}: source differs from verified backup"
            )
    if not set(current.directories).issubset(inventory.directories):
        raise MigrationInputError("source directories differ from verified backup")
    if session.manifest.cleanup_phase == "pending":
        removable = {
            f"Database/{state.source.relative_to(source.database_path).as_posix()}"
            for state in session.manifest.files
            if state.operation == "move_labber"
            and state.phase in ("published", "source_removed")
        }
        if (
            set(inventory.hashes) - set(current.hashes) - removable
            or current.directories != inventory.directories
        ):
            raise MigrationInputError("source disappeared since verified backup")
    return inventory


def _ready(session: MigrationSession) -> bool:
    manifest = session.manifest
    return (
        manifest.parameters_complete
        and manifest.data_complete
        and not manifest.report.failures
        and not any(
            item.location == "point figures" for item in manifest.report.pending
        )
        and all(
            (
                state.phase == "source_removed"
                if state.operation == "move_labber"
                else state.phase == "published"
            )
            and (state.operation != "native" or state.native_validated)
            for state in manifest.files
        )
    )


def _retained_pending(
    session: MigrationSession, backup: MigrationBackup
) -> tuple[PendingItem, ...]:
    pending: list[PendingItem] = []
    for item in session.manifest.report.pending:
        path = item.source
        for label, root in _source_roots(session.manifest.report.source).items():
            if path.is_relative_to(root):
                path = backup.path / label / path.relative_to(root)
                break
        pending.append(
            replace(
                item,
                source=path,
                reason=item.reason.replace(
                    "File remains at its original source",
                    "File retained in verified backup",
                ),
            )
        )
    return tuple(pending)


def finish_cleanup(session: MigrationSession) -> None:
    """Remove only fully backed-up sources after both migration parts complete.

    Permanent pending is retained at backup paths. Deferred point figures and
    incomplete publications block cleanup. Verify all remaining source bytes
    before authorizing deletion, checkpoint cleaning, then unlink verified files
    and remove empty directories. Resume after interruption safely repeats this
    procedure without requiring already removed sources. dry_run never deletes.
    """
    if session.dry_run or not _ready(session):
        return
    inventory = validate_backup_sources(session)
    backup = session.manifest.backup
    if backup is None:
        raise MigrationInputError("manifest: verified backup is required")
    source = session.manifest.report.source
    if session.manifest.cleanup_phase == "complete":
        if any(root.exists() for root in _source_roots(source).values()):
            raise MigrationInputError("source tree reappeared after completed cleanup")
        return
    session.manifest = replace(session.manifest, cleanup_phase="cleaning")
    session.checkpoint()
    for relative, digest in inventory.hashes.items():
        label, name = relative.split("/", 1)
        path = _source_roots(source)[label] / name
        if path.exists():
            if path.is_symlink() or file_hash(path) != digest:
                raise MigrationInputError(f"{path}: source changed during cleanup")
            path.unlink()
    for relative in sorted(
        inventory.directories, key=lambda value: len(Path(value).parts), reverse=True
    ):
        label, _, name = relative.partition("/")
        path = _source_roots(source)[label] / name
        if path.exists():
            path.rmdir()
    session.manifest = replace(
        session.manifest,
        cleanup_phase="complete",
        report=replace(
            session.manifest.report, pending=_retained_pending(session, backup)
        ),
    )
    session.checkpoint()
