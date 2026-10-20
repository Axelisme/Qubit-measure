"""Complete external source backups and hash-guarded, resumable source cleanup."""

import hashlib
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from .errors import MigrationInputError
from .models import (
    ExistingItem,
    MigrationBackup,
    MigrationDifference,
    MigrationRequest,
    MigrationSource,
    PendingItem,
)
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

    @property
    def summary(self) -> str:
        """Return file/directory counts and a typed-inventory SHA256 for diagnostics.

        This hashes the complete listing in memory, not files on disk. It does
        not verify the backup or change it. The digest distinguishes differing
        listings without printing every source path in a conflict message.
        """
        digest = hashlib.sha256(_INVENTORY.dump_json(self)).hexdigest()
        return f"{len(self.hashes)} files, {len(self.directories)} directories, sha256={digest}"


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


def inventory_paths(roots: Mapping[str, Path]) -> BackupInventory:
    """Snapshot complete regular directory trees under caller-supplied labels.

    roots maps nonoverlapping labels to exact directory paths. Return SHA256
    hashes and every directory under label/relative-path, including empty roots.
    No writes occur. Reject missing roots, symlinks and non-regular files with
    MigrationInputError; read errors propagate. This listing does not authorize
    overwrites, adoption or cleanup and does not coordinate concurrent writers.
    """
    return _inventory(roots)


def snapshot_sources(source: MigrationSource) -> BackupInventory:
    """Read complete live source hashes and directories without writing anything.

    source supplies the selected result/Database trees. Return result/Database
    relative labels, including empty roots. Reject missing roots, symlinks and
    non-regular files with MigrationInputError; read errors propagate. A snapshot
    is a baseline, not backup verification or cleanup authorization.
    """
    return _inventory(_source_roots(source))


def create_backup(
    path: Path, source: MigrationSource, *, dry_run: bool
) -> MigrationBackup | None:
    """Copy complete sources into a shared root and publish two verified hash lists.

    path is a validated external backup root; source supplies complete live trees.
    Verify every existing file/directory and inventory against those trees before
    any write. Matching files/lists are reused unchanged, missing items copied.
    Return verified inventories/source hashes and typed skipped items. Differences
    raise MigrationInputError with located comparisons, without overwriting or
    filling gaps. Symlinks and non-regular files are always refused. Unrelated
    chip/qubit backups remain untouched. dry_run performs the same preflight,
    writes nothing and returns None. I/O failures propagate; incomplete backups
    never authorize conversion/cleanup. No concurrent-writer guarantee is made.
    """
    mirrors = _mirror_roots(path, source)
    records = backup_records_path(path, source)
    for target in (*mirrors.values(), records):
        _reject_symlinks(target)
    if records.exists() and not records.is_dir():
        raise MigrationInputError(
            "backup records conflict",
            differences=(
                MigrationDifference(
                    source=source.result_path,
                    destination=records,
                    comparison="file-type",
                    expected="directory",
                    actual="non-directory",
                ),
            ),
        )
    inventory = snapshot_sources(source)
    skipped = _preflight_backup(path, source, inventory)
    if dry_run:
        return None
    for relative in inventory.directories:
        label, _, name = relative.partition("/")
        (mirrors[label] / name).mkdir(parents=True, exist_ok=True)
    for relative in inventory.hashes:
        label, name = relative.split("/", 1)
        target = mirrors[label] / name
        if not target.exists():
            shutil.copy2(_source_roots(source)[label] / name, target)
    if (
        _inventory(mirrors) != inventory
        or _inventory(_source_roots(source)) != inventory
    ):
        raise MigrationInputError(
            f"{path}: backup verification failed or sources changed"
        )
    records.mkdir(parents=True, exist_ok=True)
    inventory_hashes: dict[str, str] = {}
    for label, mirror in mirrors.items():
        inventory_path = records / f"{label}.sha256"
        mirror_label = mirror.relative_to(path).as_posix()
        if not inventory_path.exists():
            with inventory_path.open("xb") as stream:
                stream.write(
                    _INVENTORY.dump_json(_inventory({mirror_label: mirror}), indent=2)
                )
        inventory_hashes[label] = file_hash(inventory_path)
    backup = MigrationBackup(
        path=path,
        inventory_hashes=inventory_hashes,
        source_hashes=_absolute_hashes(inventory, source),
        skipped=skipped,
    )
    verify_backup(backup, source)
    return backup


def _compare_inventory(
    path: Path, source: Path, expected: BackupInventory
) -> ExistingItem | MigrationDifference:
    actual = "non-regular inventory file"
    if path.is_file():
        try:
            inventory = _INVENTORY.validate_json(path.read_bytes(), strict=True)
        except ValidationError:
            actual = "invalid JSON or backup-inventory shape"
        else:
            if inventory == expected:
                return ExistingItem(
                    source=source,
                    destination=path,
                    comparison="backup-inventory",
                    summary=file_hash(path),
                )
            actual = inventory.summary
    return MigrationDifference(
        source=source,
        destination=path,
        comparison="backup-inventory",
        expected=expected.summary,
        actual=actual,
    )


def _preflight_backup(
    path: Path, source: MigrationSource, expected: BackupInventory
) -> tuple[ExistingItem, ...]:
    mirrors = _mirror_roots(path, source)
    sources = _source_roots(source)
    differences: list[MigrationDifference] = []
    skipped: list[ExistingItem] = []
    for label, mirror in mirrors.items():
        if mirror.exists() and not mirror.is_dir():
            differences.append(
                MigrationDifference(
                    source=sources[label],
                    destination=mirror,
                    comparison="file-type",
                    expected="directory",
                    actual="non-directory",
                )
            )
            continue
        current = _inventory({label: mirror}, missing_ok=True)
        for relative in sorted(set(current.hashes) | set(current.directories)):
            _, _, name = relative.partition("/")
            actual = current.hashes.get(relative, "directory")
            wanted = expected.hashes.get(relative)
            if wanted is None:
                wanted = "directory" if relative in expected.directories else "absent"
            if actual != wanted:
                differences.append(
                    MigrationDifference(
                        source=sources[label] / name,
                        destination=mirror / name,
                        comparison="sha256"
                        if relative in current.hashes and relative in expected.hashes
                        else "file-type",
                        expected=wanted,
                        actual=actual,
                    )
                )
            elif relative in current.hashes:
                skipped.append(
                    ExistingItem(
                        source=sources[label] / name,
                        destination=mirror / name,
                        comparison="sha256",
                        summary=actual,
                    )
                )
        inventory_path = backup_records_path(path, source) / f"{label}.sha256"
        _reject_symlinks(inventory_path)
        if inventory_path.exists():
            mirror_label = mirror.relative_to(path).as_posix()
            wanted_inventory = BackupInventory(
                hashes={
                    f"{mirror_label}{key.removeprefix(label)}": value
                    for key, value in expected.hashes.items()
                    if key.startswith(f"{label}/")
                },
                directories=tuple(
                    f"{mirror_label}{key.removeprefix(label)}"
                    for key in expected.directories
                    if key == label or key.startswith(f"{label}/")
                ),
            )
            comparison = _compare_inventory(
                inventory_path, sources[label], wanted_inventory
            )
            if isinstance(comparison, MigrationDifference):
                differences.append(comparison)
            else:
                skipped.append(comparison)
    if differences:
        raise MigrationInputError(
            "backup differs from source", differences=tuple(differences)
        )
    return tuple(skipped)


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
        and not any(item.location == "point files" for item in manifest.report.pending)
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

    Permanent pending is retained at backup paths. Deferred point files and
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
