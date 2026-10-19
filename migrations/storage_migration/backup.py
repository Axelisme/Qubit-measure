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

    hashes maps each caller-supplied root label plus relative file path to
    lowercase SHA256. On disk the labels are result/<chip>/<qubit> and
    Database/<chip>/<qubit>; cleanup uses result and Database labels internally.
    directories records every directory, including empty source roots.
    """

    hashes: Mapping[str, str]
    directories: tuple[str, ...]


_INVENTORY = TypeAdapter(BackupInventory)


def backup_location(request: MigrationRequest, source: MigrationSource) -> Path:
    """Resolve an external <repo-name>-backup root without writes or overlaps.

    request.repo_root defaults to the tool checkout. Reject a backup within that
    repo, enclosing either source, or overlapping either destination. Reports
    cannot write into the logical mirrors.
    Existence is checked separately so resume can verify the owned backup.
    """
    repo = (request.repo_root or Path(__file__).resolve().parents[2]).resolve()
    candidate = (request.backup_parent or repo.parent) / f"{repo.name}-backup"
    _reject_symlinks(candidate)
    path = candidate.resolve()
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
    if request.report_path is not None:
        validate_backup_report_path(path, request.report_path)
    return path


def _reject_symlinks(path: Path) -> None:
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink():
            raise MigrationInputError(f"{ancestor}: backup symlink conflict")


def _mirror_roots(path: Path, source: MigrationSource) -> dict[str, Path]:
    relative = Path(source.result_path.parent.name) / source.result_path.name
    return {label: path / label / relative for label in ("result", "Database")}


def backup_records_path(path: Path, source: MigrationSource) -> Path:
    """Return this source's records directory under the shared backup root.

    path is the validated backup root; source names the selected chip/qubit.
    This only computes a location, never creates or adopts existing records.
    """
    return (
        path
        / "migration-records"
        / source.result_path.parent.name
        / source.result_path.name
    )


def validate_backup_report_path(path: Path, report_path: Path) -> None:
    """Reject a report inside either logical mirror, without writing anything.

    path is the shared backup root. Reports may use its non-mirror locations or
    an independent external path. Symlink ancestors are unsupported.
    """
    _reject_symlinks(report_path)
    if any(
        report_path.resolve().is_relative_to(path / label)
        for label in ("result", "Database")
    ):
        raise MigrationInputError(
            f"{report_path}: report cannot write into backup mirrors"
        )


def backup_source_path(path: Path, source: MigrationSource, original: Path) -> Path:
    """Locate an original source file in the logical backup mirror, without I/O.

    path is the shared backup root; source owns the original result/Database
    trees. Return original unchanged if it belongs to neither source tree.
    """
    mirrors = _mirror_roots(path, source)
    for label, root in _source_roots(source).items():
        if original.is_relative_to(root):
            return mirrors[label] / original.relative_to(root)
    return original


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
    """Copy complete sources into a shared root and publish two verified hash lists.

    Refuse either existing chip/qubit mirror or records directory, symlinks and
    non-regular files. Unrelated backups remain untouched. dry_run validates but
    writes nothing and returns None. Failures retain partial backups for operator
    review, never grant migration permission or remove the shared root.
    """
    mirrors = _mirror_roots(path, source)
    records = backup_records_path(path, source)
    for target in (*mirrors.values(), records):
        _reject_symlinks(target)
        if target.exists():
            raise FileExistsError(target)
    inventory = _inventory(_source_roots(source))
    if dry_run:
        return None
    for label, root in _source_roots(source).items():
        mirrors[label].parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(root, mirrors[label], copy_function=shutil.copy2)
    if (
        _inventory(mirrors) != inventory
        or _inventory(_source_roots(source)) != inventory
    ):
        raise MigrationInputError(
            f"{path}: backup verification failed or sources changed"
        )
    records.mkdir(parents=True, exist_ok=False)
    inventory_hashes: dict[str, str] = {}
    for label, mirror in mirrors.items():
        inventory_path = records / f"{label}.sha256"
        mirror_label = mirror.relative_to(path).as_posix()
        with inventory_path.open("xb") as stream:
            stream.write(
                _INVENTORY.dump_json(_inventory({mirror_label: mirror}), indent=2)
            )
        inventory_hashes[label] = file_hash(inventory_path)
    backup = MigrationBackup(
        path=path,
        inventory_hashes=inventory_hashes,
        source_hashes=_absolute_hashes(inventory, source),
    )
    verify_backup(backup, source)
    return backup


def verify_backup(backup: MigrationBackup, source: MigrationSource) -> BackupInventory:
    """Verify the bound inventory and every backup file, independent of old sources.

    source identifies this chip/qubit's mirrors. Return their complete typed listing.
    Reject changed lists, hashes, extra files within these mirrors,
    missing directories and symlinks with MigrationInputError; I/O propagates.
    No backup is rewritten, even when migration has already moved source files.
    """
    path = backup.path
    mirrors = _mirror_roots(path, source)
    records = backup_records_path(path, source)
    if set(backup.inventory_hashes) != set(mirrors):
        raise MigrationInputError(f"{path}: backup inventory identities differ")
    hashes: dict[str, str] = {}
    directories: list[str] = []
    for label, mirror in mirrors.items():
        inventory_path = records / f"{label}.sha256"
        _reject_symlinks(inventory_path)
        _reject_symlinks(mirror)
        if file_hash(inventory_path) != backup.inventory_hashes[label]:
            raise MigrationInputError(f"{path}: backup inventory hash changed")
        try:
            inventory = _INVENTORY.validate_json(
                inventory_path.read_bytes(), strict=True
            )
        except ValidationError as exc:
            raise MigrationInputError(f"{path}: invalid backup inventory") from exc
        mirror_label = mirror.relative_to(path).as_posix()
        if _inventory({mirror_label: mirror}) != inventory:
            raise MigrationInputError(f"{path}: backup contents differ from inventory")
        hashes.update(
            {
                f"{label}{name.removeprefix(mirror_label)}": digest
                for name, digest in inventory.hashes.items()
            }
        )
        directories.extend(
            f"{label}{name.removeprefix(mirror_label)}"
            for name in inventory.directories
        )
    return BackupInventory(hashes=hashes, directories=tuple(sorted(directories)))


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
    source = session.manifest.report.source
    inventory = verify_backup(backup, source)
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
        path = backup_source_path(
            backup.path, session.manifest.report.source, item.source
        )
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
