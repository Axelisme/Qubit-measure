"""Offline filename upgrades without parsing or changing document bytes."""

from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path

from filelock import FileLock, Timeout

from zcu_tools.resources.document_names import (
    LEGACY_DOCUMENT_NAMES,
    LEGACY_SETUP_FILENAME,
    PARAMETERS_FILENAME,
)
from zcu_tools.resources.file_lock import lock_path_for
from zcu_tools.resources.upgrade_backup import upgrade_backup_path


def rename_document_files(
    paths: Sequence[Path], *, apply: bool = False, backup_root: Path | None = None
) -> int:
    """Print filename changes for exact old files or entry roots; return 0/1.

    Entry roots must contain old setup or canonical parameters. Only that root
    and direct points/* directories are examined. No YAML is parsed. Dry-run
    creates no files or locks. Apply backs up original bytes using the shared
    external backup policy, then links the new name exclusively and unlinks the
    old name under both runtime locks. Existing targets are never overwritten.
    OSError/ValueError produce diagnostics and return 1. No multi-file or
    power-loss transaction is provided; stop other writers before applying.
    """
    try:
        pairs = _rename_pairs(paths)
        targets: set[Path] = set()
        for source, destination in pairs:
            if (
                source.is_symlink()
                or destination.exists()
                or destination.is_symlink()
                or destination in targets
            ):
                raise ValueError(
                    f"{source} -> {destination}: conflicting names or symlink"
                )
            targets.add(destination)
        for source, destination in pairs:
            print(f"{source} -> {destination}: {'apply' if apply else 'dry-run'}")
            if not apply:
                continue
            backup = upgrade_backup_path(source, backup_root=backup_root)
            with ExitStack() as stack:
                for path in sorted((source, destination)):
                    stack.enter_context(FileLock(str(lock_path_for(path)), timeout=5))
                original = source.read_bytes()
                backup.parent.mkdir(parents=True, exist_ok=True)
                with backup.open("xb") as stream:
                    stream.write(original)
                print(f"{source}: backup: {backup}")
                destination.hardlink_to(source)
                source.unlink()
        return 0
    except (OSError, ValueError, Timeout) as error:
        print(f"filename upgrade rejected: {error}")
        return 1


def _rename_pairs(paths: Sequence[Path]) -> tuple[tuple[Path, Path], ...]:
    pairs: dict[Path, Path] = {}
    names = dict(LEGACY_DOCUMENT_NAMES)
    for requested in paths:
        path = requested.absolute()
        if not path.is_dir():
            if path.name not in names:
                raise ValueError(f"{path}: expected an exact legacy document filename")
            path.stat()
            pairs[path] = path.with_name(names[path.name])
            continue
        if not any(
            (path / name).is_file()
            for name in (LEGACY_SETUP_FILENAME, PARAMETERS_FILENAME)
        ):
            raise ValueError(f"{path}: not an entry directory")
        directories = [
            path,
            *sorted(child for child in (path / "points").glob("*") if child.is_dir()),
        ]
        for directory in directories:
            for old, new in LEGACY_DOCUMENT_NAMES:
                source = directory / old
                if source.is_file():
                    pairs[source] = directory / new
    return tuple(pairs.items())
