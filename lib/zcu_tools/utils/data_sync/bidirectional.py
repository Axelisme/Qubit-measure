"""Plan, copy and verify a single Labdata prefix in both directions."""

from __future__ import annotations

import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath

from . import rclone
from .baseline import BaselineEntry, BaselineStore, verified_entries, verified_entry
from .inventory import (
    DestinationPaths,
    FileInventory,
    ListedFile,
    list_case_insensitive_destination,
    list_files,
    mtime_order,
)


@dataclass(frozen=True)
class SyncPair:
    """Validated endpoints and their shared Labdata baseline."""

    local: str  # Absolute physical prefix root.
    remote: str  # Named rclone path, including the prefix.
    store: BaselineStore  # Record for exactly these two endpoints.


class _Action(StrEnum):
    UPLOAD = "Upload"
    DOWNLOAD = "Download"
    SKIP = "Skip"
    RETIRE = "Retire"
    CONFLICT = "Conflict"


@dataclass(frozen=True)
class _Step:
    action: _Action
    reason: str


def sync_pair(
    binary: Path,
    config_dir: Path,
    pair: SyncPair,
    *,
    dry_run: bool,
    copy_tuning: list[str],
) -> int:
    """Print the pair's plan and, unless dry-run or blocked, copy and verify it.

    Use the verified rclone binary, config_dir/rclone.conf and copy performance
    and reporting flags only. Return the copy exit status on copy failure, or 1
    for a blocked plan or unverified transfer. Raise on inventory, destination
    capability or baseline errors. Dry-run reads but does not change data or
    the baseline.
    """
    previous = pair.store.load() or {}
    local, remote = _list_pair(binary, config_dir, pair)
    plan = _plan(local, remote, previous)
    for action, root in ((_Action.UPLOAD, pair.remote), (_Action.DOWNLOAD, pair.local)):
        writes = [path for path, step in plan.items() if step.action == action]
        if writes and rclone.case_insensitive(binary, config_dir, root):
            destination = list_case_insensitive_destination(binary, config_dir, root)
            _flag_case_collisions(plan, writes, destination)
    _print_plan(pair, plan)
    if any(step.action == _Action.CONFLICT for step in plan.values()):
        return 1
    transfers = {
        action: [path for path, step in plan.items() if step.action == action]
        for action in (_Action.UPLOAD, _Action.DOWNLOAD)
    }
    batches = {action: _encode_paths(paths) for action, paths in transfers.items()}
    if dry_run:
        return 0

    code = _copy_batches(binary, config_dir, pair, copy_tuning, batches)
    if code:
        return code
    copied = set(transfers[_Action.UPLOAD]) | set(transfers[_Action.DOWNLOAD])
    try:
        if copied:
            local, remote = _list_pair(binary, config_dir, pair)
        verified = verified_entries(local, remote)
        unverified = copied - verified.keys()
        _update_baseline(pair.store, previous, plan, verified, retire=not unverified)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            "Files may have been copied, but the sync baseline was not updated: "
            f"{pair.store.path}. {exc}"
        ) from exc
    for path in sorted(unverified):
        print(
            f"Could not verify both copies of {path!r}; baseline kept.", file=sys.stderr
        )
    return int(bool(unverified))


def _update_baseline(
    store: BaselineStore,
    previous: dict[str, BaselineEntry],
    plan: dict[str, _Step],
    verified: dict[str, BaselineEntry],
    *,
    retire: bool,
) -> None:
    updated = previous | {
        path: verified[path]
        for path, step in plan.items()
        if step.action != _Action.RETIRE and path in verified
    }
    if retire:
        for path, step in plan.items():
            if step.action == _Action.RETIRE:
                del updated[path]
    if updated != previous:
        store.save(updated)


def _list_pair(
    binary: Path, config_dir: Path, pair: SyncPair
) -> tuple[FileInventory, FileInventory]:
    return (
        list_files(binary, config_dir, pair.local, allow_missing=True),
        list_files(binary, config_dir, pair.remote, allow_missing=True),
    )


def _plan(
    local: FileInventory, remote: FileInventory, previous: dict[str, BaselineEntry]
) -> dict[str, _Step]:
    duplicates = local.duplicates | remote.duplicates
    routes: dict[str, _Step] = {}
    for path in sorted(
        local.files.keys() | remote.files.keys() | duplicates | previous.keys()
    ):
        if path in duplicates:
            step = _Step(_Action.CONFLICT, "duplicate path in inventory")
        else:
            step = _plan_path(
                local.files.get(path), remote.files.get(path), previous.get(path)
            )
        if step.action == _Action.UPLOAD:
            step = _check_destination(path, step, remote)
        elif step.action == _Action.DOWNLOAD:
            step = _check_destination(path, step, local)
        routes[path] = step
    return routes


def _flag_case_collisions(
    plan: dict[str, _Step], writes: list[str], target: DestinationPaths
) -> None:
    files = _case_index(target.files | set(writes))
    directories = _case_index(target.directories)
    for path in writes:
        key = path.casefold()
        others = (files[key] - {path}) | directories.get(key, set())
        for parent in PurePosixPath(path).parents:
            others.update(files.get(parent.as_posix().casefold(), set()))
        if others:
            names = ", ".join(map(repr, sorted(others)))
            plan[path] = _Step(
                _Action.CONFLICT, f"case-insensitive destination collision with {names}"
            )


def _case_index(paths: set[str]) -> dict[str, set[str]]:
    names: dict[str, set[str]] = defaultdict(set)
    for path in paths:
        names[path.casefold()].add(path)
    return names


def _check_destination(path: str, step: _Step, target: FileInventory) -> _Step:
    if path in target.directories:
        return _Step(_Action.CONFLICT, "a directory occupies the destination path")
    for parent in PurePosixPath(path).parents:
        if parent.as_posix() in target.files or parent.as_posix() in target.root_files:
            return _Step(
                _Action.CONFLICT,
                f"a file occupies destination directory {str(parent)!r}",
            )
    return step


def _plan_path(
    local: ListedFile | None, remote: ListedFile | None, previous: BaselineEntry | None
) -> _Step:
    if local is None and remote is None:
        return _Step(_Action.RETIRE, "deleted on both endpoints; retire baseline")
    if local is None:
        return (
            _Step(_Action.CONFLICT, "deleted in Labdata since baseline")
            if previous is not None
            else _Step(_Action.DOWNLOAD, "only on cloud")
        )
    if remote is None:
        return (
            _Step(_Action.CONFLICT, "deleted on cloud since baseline")
            if previous is not None
            else _Step(_Action.UPLOAD, "only in Labdata")
        )
    if previous is not None:
        return _baseline_step(local, remote, previous)
    return _initial_step(local, remote)


def _initial_step(local: ListedFile, remote: ListedFile) -> _Step:
    left, right = mtime_order(local.mtime), mtime_order(remote.mtime)
    if left > right:
        return _Step(_Action.UPLOAD, "newer in Labdata")
    if right > left:
        return _Step(_Action.DOWNLOAD, "newer on cloud")
    return _compare_content(local, remote, "equal modification times")


def _baseline_step(
    local: ListedFile, remote: ListedFile, entry: BaselineEntry
) -> _Step:
    if any(
        file.size_bytes is None and file.mtime == entry[side]["mtime"]
        for side, file in (("local", local), ("remote", remote))
    ):
        return _Step(_Action.CONFLICT, "size unavailable for baseline comparison")
    local_changed = local.stamp() != entry["local"]
    remote_changed = remote.stamp() != entry["remote"]
    if local_changed and remote_changed:
        return _compare_content(local, remote, "changed on both endpoints")
    if local_changed:
        return _Step(_Action.UPLOAD, "changed only in Labdata since baseline")
    if remote_changed:
        return _Step(_Action.DOWNLOAD, "changed only on cloud since baseline")
    return _Step(_Action.SKIP, "unchanged since baseline")


def _compare_content(local: ListedFile, remote: ListedFile, reason: str) -> _Step:
    if verified_entry(local, remote) is not None:
        return _Step(_Action.SKIP, "matching contents")
    return _Step(_Action.CONFLICT, f"{reason}; contents differ or cannot be verified")


def _print_plan(pair: SyncPair, plan: dict[str, _Step]) -> None:
    print(f"Sync: {pair.local} <-> {pair.remote}")
    for path, step in plan.items():
        label = _Action.SKIP if step.action == _Action.RETIRE else step.action
        print(f"{label}: {path!r} ({step.reason})")
    if not plan:
        print("No data files to sync.")
    sys.stdout.flush()


def _encode_paths(paths: list[str]) -> bytes:
    # Rclone v1.75.2 accepts NUL-separated literal paths without line processing.
    chunks: list[bytes] = []
    for path in paths:
        try:
            chunks.append(path.encode("utf-8") + b"\0")
        except UnicodeEncodeError as exc:
            raise ValueError(f"Cannot encode sync path {path!r} as UTF-8.") from exc
    return b"".join(chunks)


def _copy_batches(
    binary: Path,
    config_dir: Path,
    pair: SyncPair,
    tuning: list[str],
    batches: dict[_Action, bytes],
) -> int:
    if not any(batches.values()):
        return 0
    with tempfile.TemporaryDirectory() as folder:
        commands: list[list[str]] = []
        for action, source, target in (
            (_Action.UPLOAD, pair.local, pair.remote),
            (_Action.DOWNLOAD, pair.remote, pair.local),
        ):
            if not batches[action]:
                continue
            manifest = Path(folder) / action.value
            manifest.write_bytes(batches[action])
            commands.append(
                [
                    "copy",
                    source,
                    target,
                    *tuning,
                    "--ignore-times",
                    "--no-traverse",
                    "--no-unicode-normalization",
                    "--files-from0",
                    str(manifest),
                ]
            )
        for command in commands:
            result = rclone.run_cli(binary, config_dir, command)
            if result.returncode:
                return result.returncode
    return 0
