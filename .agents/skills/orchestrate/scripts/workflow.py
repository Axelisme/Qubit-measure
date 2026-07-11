#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

QUEUE_VERSION = 1
STATE_VERSION = 2
QUEUE_ACTIONS = {"preview", "final"}
QUEUE_STATUSES = {"queued", "merging", "blocked"}
BLOCKED_KINDS = {
    "manual",
    "integration_refresh_failed",
    "preview_abort_failed",
    "merge_target_preflight_failed",
    "preview_target_stale",
    "preview_merge_failed",
    "preview_postcondition_failed",
    "final_fast_forward_failed",
    "final_postcondition_failed",
}
TASK_STATUSES = {"active", "reviewing", "merge_preview", "blocked"}
LANE_ROLES = {"lane", "integration"}
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
OBJECT_ID_PATTERN = re.compile(r"^[0-9A-Fa-f]{7,40}$")


class OrchestrateError(RuntimeError):
    def __init__(self, message: str, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def state_path(root: Path) -> Path:
    return root / ".agent_state" / "worktrees" / "state.json"


def queue_path(root: Path) -> Path:
    return root / ".agent_state" / "worktrees" / "merge_queue.json"


def now_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def validate_id(value: str, label: str) -> None:
    if not ID_PATTERN.fullmatch(value):
        raise OrchestrateError(
            f"{label} must match {ID_PATTERN.pattern}: {value!r}",
            40,
        )


def validate_status(value: str, label: str) -> None:
    if value not in TASK_STATUSES:
        raise OrchestrateError(
            f"{label} must be one of {sorted(TASK_STATUSES)}: {value!r}",
            40,
        )


def validate_role(value: str) -> None:
    if value not in LANE_ROLES:
        raise OrchestrateError(
            f"role must be one of {sorted(LANE_ROLES)}: {value!r}",
            40,
        )


def validate_action(value: str, exit_code: int = 40) -> None:
    if value not in QUEUE_ACTIONS:
        raise OrchestrateError(
            f"action must be one of {sorted(QUEUE_ACTIONS)}: {value!r}",
            exit_code,
        )


def validate_queue_status(value: str, exit_code: int = 40) -> None:
    if value not in QUEUE_STATUSES:
        raise OrchestrateError(
            f"queue status must be one of {sorted(QUEUE_STATUSES)}: {value!r}",
            exit_code,
        )


def validate_blocked_kind(value: str, exit_code: int = 40) -> None:
    if value not in BLOCKED_KINDS:
        raise OrchestrateError(
            f"blocked_kind must be one of {sorted(BLOCKED_KINDS)}: {value!r}",
            exit_code,
        )


def normalize_root(raw_root: str) -> Path:
    return Path(raw_root).expanduser().resolve()


def normalize_repo_path(root: Path, raw_path: str) -> str:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = root / path
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise OrchestrateError(
            f"path must be inside repo root {root}: {raw_path}",
            40,
        ) from exc


def normalize_scope(raw_scope: str) -> str:
    scope = raw_scope.replace("\\", "/").removeprefix("./")
    if not scope:
        raise OrchestrateError("write scope must not be empty", 40)
    return scope


def normalize_scope_list(raw_scopes: list[str]) -> list[str]:
    scopes: list[str] = []
    seen: set[str] = set()
    for raw_scope in raw_scopes:
        scope = normalize_scope(raw_scope)
        if scope not in seen:
            scopes.append(scope)
            seen.add(scope)
    return scopes


def parse_ignored_input(raw: str) -> dict[str, str]:
    mode, sep, path = raw.partition(":")
    if not sep or mode not in {"copy", "reference", "omit"}:
        raise OrchestrateError(
            "--ignored-input must use copy:<path>, reference:<path>, or omit:<label>",
            40,
        )
    return {"mode": mode, "path": path}


def load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise OrchestrateError(f"{path} is not valid JSON: {exc}", 10) from exc
    if not isinstance(data, dict):
        raise OrchestrateError(f"{path} must contain a JSON object", 10)
    return data


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


@contextmanager
def file_lock(target: Path, timeout: float) -> Iterator[None]:
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_name(f"{target.name}.lock")
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(f"pid={os.getpid()}\n")
            break
        except FileExistsError as exc:
            if time.monotonic() >= deadline:
                raise OrchestrateError(
                    f"timed out waiting for lock {lock_path}",
                    30,
                ) from exc
            time.sleep(0.1)
    try:
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def empty_state() -> dict[str, Any]:
    return {"version": STATE_VERSION, "tasks": {}}


def load_state(root: Path) -> dict[str, Any]:
    data = load_json(state_path(root), empty_state())
    validate_state(data)
    return data


def save_state(root: Path, data: dict[str, Any]) -> None:
    validate_state(data)
    atomic_write_json(state_path(root), data)


@contextmanager
def locked_state(root: Path, timeout: float) -> Iterator[dict[str, Any]]:
    path = state_path(root)
    with file_lock(path, timeout):
        data = load_state(root)
        yield data
        save_state(root, data)


def validate_state(data: dict[str, Any]) -> None:
    if data.get("version") != STATE_VERSION:
        raise OrchestrateError(f"state version must be {STATE_VERSION}", 10)
    tasks = data.get("tasks")
    if not isinstance(tasks, dict):
        raise OrchestrateError("state.tasks must be an object", 10)
    for task_id, task in tasks.items():
        validate_id(task_id, "task-id")
        if not isinstance(task, dict):
            raise OrchestrateError(f"task {task_id} must be an object", 10)
        validate_status(str(task.get("status")), f"task {task_id} status")
        for field in ("base_branch", "base_commit", "integration_branch"):
            if not isinstance(task.get(field), str) or not task[field]:
                raise OrchestrateError(f"task {task_id}.{field} must be a string", 10)
        worktrees = task.get("worktrees")
        if not isinstance(worktrees, dict):
            raise OrchestrateError(f"task {task_id}.worktrees must be an object", 10)
        for lane_id, lane in worktrees.items():
            validate_id(lane_id, "lane-id")
            if not isinstance(lane, dict):
                raise OrchestrateError(
                    f"task {task_id} lane {lane_id} must be an object",
                    10,
                )
            validate_status(
                str(lane.get("status")),
                f"task {task_id} lane {lane_id} status",
            )
            validate_role(str(lane.get("role")))
            for field in ("branch", "worktree_path", "reports_dir"):
                if not isinstance(lane.get(field), str) or not lane[field]:
                    raise OrchestrateError(
                        f"task {task_id} lane {lane_id}.{field} must be a string",
                        10,
                    )
            if not isinstance(lane.get("write_scope"), list):
                raise OrchestrateError(
                    f"task {task_id} lane {lane_id}.write_scope must be a list",
                    10,
                )
            for index, scope in enumerate(lane["write_scope"]):
                if not isinstance(scope, str) or not scope:
                    raise OrchestrateError(
                        f"task {task_id} lane {lane_id}.write_scope[{index}] "
                        "must be a non-empty string",
                        10,
                    )
            if not isinstance(lane.get("ignored_inputs"), list):
                raise OrchestrateError(
                    f"task {task_id} lane {lane_id}.ignored_inputs must be a list",
                    10,
                )


def empty_queue() -> dict[str, Any]:
    return {"version": QUEUE_VERSION, "queue": []}


def validate_queue(data: dict[str, Any]) -> None:
    if data.get("version") != QUEUE_VERSION:
        raise OrchestrateError(f"queue version must be {QUEUE_VERSION}", 10)
    entries = data.get("queue")
    if not isinstance(entries, list):
        raise OrchestrateError("queue.queue must be a list", 10)
    seen_task_ids: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise OrchestrateError(f"queue entry {index} must be an object", 10)
        for field in ("task_id", "branch", "base_branch", "requested_by"):
            if not isinstance(entry.get(field), str) or not entry[field]:
                raise OrchestrateError(
                    f"queue entry {index}.{field} must be a string",
                    10,
                )
        validate_id(str(entry["task_id"]), "task-id")
        if entry["task_id"] in seen_task_ids:
            raise OrchestrateError(
                f"queue contains duplicate task_id {entry['task_id']}",
                10,
            )
        seen_task_ids.add(entry["task_id"])
        validate_action(str(entry.get("action")), 10)
        status = str(entry.get("status"))
        validate_queue_status(status, 10)
        blocked_kind = entry.get("blocked_kind")
        if blocked_kind is not None:
            if not isinstance(blocked_kind, str):
                raise OrchestrateError(
                    f"queue entry {index}.blocked_kind must be a string or null",
                    10,
                )
            validate_blocked_kind(blocked_kind, 10)
        if status != "blocked" and blocked_kind is not None:
            raise OrchestrateError(
                f"queue entry {index}.blocked_kind must be null when status is {status}",
                10,
            )
        if not isinstance(entry.get("enqueued_at"), str):
            raise OrchestrateError(
                f"queue entry {index}.enqueued_at must be a string",
                10,
            )
        for field in (
            "started_at",
            "note",
            "token",
            "target_commit",
            "base_head",
            "main_worktree",
        ):
            value = entry.get(field)
            if value is not None and not isinstance(value, str):
                raise OrchestrateError(
                    f"queue entry {index}.{field} must be a string or null",
                    10,
                )


def load_queue(root: Path) -> dict[str, Any]:
    data = load_json(queue_path(root), empty_queue())
    validate_queue(data)
    return data


def save_queue(root: Path, data: dict[str, Any]) -> None:
    validate_queue(data)
    atomic_write_json(queue_path(root), data)


def queue_entries(queue: dict[str, Any]) -> list[dict[str, Any]]:
    entries = queue["queue"]
    if not isinstance(entries, list):
        raise OrchestrateError("queue.queue must be a list", 10)
    return entries


def save_workflow(root: Path, state: dict[str, Any], queue: dict[str, Any]) -> None:
    save_state(root, state)
    save_queue(root, queue)


@contextmanager
def workflow_locks(root: Path, timeout: float) -> Iterator[None]:
    with file_lock(queue_path(root), timeout), file_lock(state_path(root), timeout):
        yield


def run_git(
    root: Path,
    *git_args: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *git_args],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise OrchestrateError(
            f"git {' '.join(git_args)} failed: {format_git_output(result)}",
            50,
        )
    return result


def format_git_output(result: subprocess.CompletedProcess[str]) -> str:
    output = "\n".join(
        part.strip() for part in (result.stdout, result.stderr) if part and part.strip()
    )
    return output or f"exit code {result.returncode}"


def git_stdout(root: Path, *git_args: str) -> str:
    return run_git(root, *git_args).stdout.strip()


def git_commit(root: Path, rev: str) -> str:
    return git_stdout(root, "rev-parse", "--verify", f"{rev}^{{commit}}")


def git_branch_exists(root: Path, branch: str) -> bool:
    return (
        run_git(
            root,
            "show-ref",
            "--verify",
            "--quiet",
            f"refs/heads/{branch}",
            check=False,
        ).returncode
        == 0
    )


def git_is_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
    return (
        run_git(
            root,
            "merge-base",
            "--is-ancestor",
            ancestor,
            descendant,
            check=False,
        ).returncode
        == 0
    )


def git_current_branch(root: Path) -> str:
    branch = git_stdout(root, "branch", "--show-current")
    if not branch:
        raise OrchestrateError(f"{root} is in detached HEAD state", 40)
    return branch


def ensure_on_base_branch(root: Path, task: dict[str, Any]) -> None:
    expected_branch = task["base_branch"]
    current_branch = git_current_branch(root)
    if current_branch != expected_branch:
        raise OrchestrateError(
            f"current branch is {current_branch}, expected base branch {expected_branch}",
            40,
        )


def git_merge_head_path(root: Path) -> Path:
    raw_path = git_stdout(root, "rev-parse", "--git-path", "MERGE_HEAD")
    path = Path(raw_path)
    if not path.is_absolute():
        path = root / path
    return path


def git_path(root: Path, name: str) -> Path:
    raw_path = git_stdout(root, "rev-parse", "--git-path", name)
    path = Path(raw_path)
    if not path.is_absolute():
        path = root / path
    return path


def git_merge_head(root: Path) -> str | None:
    path = git_merge_head_path(root)
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return None
    return lines[0].strip() or None


def ensure_no_merge_in_progress(root: Path) -> None:
    current = git_merge_head(root)
    if current is not None:
        raise OrchestrateError(
            f"worktree {root} already has a merge preview for {current}",
            40,
        )


def ensure_no_rebase_in_progress(root: Path) -> None:
    active_paths = [
        path
        for name in ("rebase-merge", "rebase-apply")
        if (path := git_path(root, name)).exists()
    ]
    if active_paths:
        raise OrchestrateError(
            f"integration worktree has a rebase in progress: {active_paths[0]}",
            40,
        )


def ensure_tracked_clean_worktree(root: Path) -> None:
    status = run_git(
        root, "status", "--porcelain", "--untracked-files=no"
    ).stdout.strip()
    if status:
        raise OrchestrateError(f"worktree {root} has tracked changes:\n{status}", 40)


def git_null_list(root: Path, *git_args: str) -> list[str]:
    output = run_git(root, *git_args).stdout
    return [item for item in output.split("\0") if item]


def paths_overlap(left: str, right: str) -> bool:
    left_parts = Path(left).parts
    right_parts = Path(right).parts
    common = min(len(left_parts), len(right_parts))
    return left_parts[:common] == right_parts[:common]


def ensure_untracked_files_do_not_overlap_merge_target(
    root: Path,
    target_commit: str,
) -> None:
    untracked_paths = git_null_list(
        root,
        "ls-files",
        "--others",
        "-z",
    )
    if not untracked_paths:
        return
    target_paths = git_null_list(
        root,
        "diff",
        "--name-only",
        "-z",
        "HEAD",
        target_commit,
    )
    collisions = sorted(
        untracked_path
        for untracked_path in untracked_paths
        if any(
            paths_overlap(untracked_path, target_path) for target_path in target_paths
        )
    )
    if collisions:
        raise OrchestrateError(
            "untracked files overlap the merge target:\n" + "\n".join(collisions),
            40,
        )


def require_task(state: dict[str, Any], task_id: str) -> dict[str, Any]:
    task = state["tasks"].get(task_id)
    if task is None:
        raise OrchestrateError(f"task {task_id} does not exist", 40)
    if not isinstance(task, dict):
        raise OrchestrateError(f"task {task_id} must be an object", 10)
    return task


def find_queue_entry(
    queue: dict[str, Any],
    task_id: str,
) -> tuple[int, dict[str, Any] | None]:
    for index, entry in enumerate(queue_entries(queue)):
        if entry["task_id"] == task_id:
            return index, entry
    return -1, None


def queue_wait_error(queue: dict[str, Any], task_id: str) -> OrchestrateError:
    entries = queue_entries(queue)
    head = entries[0]["task_id"] if entries else "<none>"
    return OrchestrateError(f"task {task_id} is queued behind {head}", 20)


def blocked_queue_head_error(entry: dict[str, Any]) -> OrchestrateError:
    return OrchestrateError(
        f"merge queue head {entry['task_id']} is blocked: {entry.get('note', '')}",
        20,
    )


def require_pinned_target(entry: dict[str, Any]) -> str:
    target = entry.get("target_commit")
    if not isinstance(target, str) or not target:
        raise OrchestrateError(
            f"queue entry for {entry['task_id']} is missing target_commit",
            40,
        )
    return target


def build_queue_entry(
    *,
    task_id: str,
    task: dict[str, Any],
    action: str,
    requested_by: str,
    target_commit: str | None,
    base_head: str | None,
    main_worktree: Path,
    note: str,
    current: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_action(action)
    return {
        "task_id": task_id,
        "branch": task["integration_branch"],
        "base_branch": task["base_branch"],
        "action": action,
        "requested_by": requested_by,
        "status": "queued",
        "blocked_kind": None,
        "enqueued_at": current["enqueued_at"] if current else now_utc(),
        "started_at": None,
        "note": note,
        "token": None,
        "target_commit": target_commit,
        "base_head": base_head,
        "main_worktree": main_worktree.as_posix(),
    }


def upsert_queue_entry(
    queue: dict[str, Any],
    entry: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    entries = queue_entries(queue)
    index, current = find_queue_entry(queue, str(entry["task_id"]))
    if current is None:
        entries.append(entry)
        return len(entries) - 1, entry
    entries[index] = entry
    return index, entry


def requeue_for_validation(
    *,
    args: argparse.Namespace,
    root: Path,
    state: dict[str, Any],
    queue: dict[str, Any],
    task: dict[str, Any],
    entry: dict[str, Any],
    target_commit: str,
    base_head: str,
    note: str,
    message: str,
) -> None:
    entry.update(
        build_queue_entry(
            task_id=args.task_id,
            task=task,
            action=args.action,
            requested_by=args.requested_by,
            target_commit=target_commit,
            base_head=base_head,
            main_worktree=root,
            note=note,
            current=entry,
        )
    )
    task["status"] = "reviewing"
    save_workflow(root, state, queue)
    raise OrchestrateError(message, 20)


def claim_entry(
    entry: dict[str, Any],
    *,
    action: str,
    requested_by: str,
    target_commit: str,
    base_head: str,
    main_worktree: Path,
    note: str,
) -> None:
    validate_action(action)
    entry.update(
        {
            "action": action,
            "requested_by": requested_by,
            "status": "merging",
            "blocked_kind": None,
            "started_at": now_utc(),
            "note": note,
            "token": uuid4().hex,
            "target_commit": target_commit,
            "base_head": base_head,
            "main_worktree": main_worktree.as_posix(),
        }
    )


def block_entry(entry: dict[str, Any], note: str, *, kind: str) -> None:
    validate_blocked_kind(kind)
    entry["status"] = "blocked"
    entry["blocked_kind"] = kind
    entry["note"] = note


def abort_preview_merge(root: Path, entry: dict[str, Any]) -> None:
    expected_target = entry.get("target_commit")
    current_target = git_merge_head(root)
    if current_target is None:
        ensure_tracked_clean_worktree(root)
    elif expected_target and current_target != expected_target:
        raise OrchestrateError(
            f"merge preview target is {current_target}, expected {expected_target}",
            50,
        )
    else:
        result = run_git(root, "merge", "--abort", check=False)
        if result.returncode != 0:
            raise OrchestrateError(
                f"git merge --abort failed: {format_git_output(result)}",
                50,
            )
    ensure_no_merge_in_progress(root)
    ensure_tracked_clean_worktree(root)
    expected_base = entry.get("base_head")
    if expected_base:
        current_head = git_commit(root, "HEAD")
        if current_head != expected_base:
            raise OrchestrateError(
                f"merge abort left HEAD at {current_head}, expected {expected_base}",
                50,
            )


def find_integration_worktree(root: Path, task: dict[str, Any]) -> Path | None:
    integration_branch = task["integration_branch"]
    lanes = list(task["worktrees"].values())
    candidates = [
        lane
        for lane in lanes
        if lane["role"] == "integration" or lane["branch"] == integration_branch
    ]
    for lane in candidates:
        path = root / lane["worktree_path"]
        if path.exists():
            return path
    return None


def read_merge_ref_snapshot(root: Path, task: dict[str, Any]) -> tuple[str, str]:
    return git_commit(root, "HEAD"), git_commit(root, task["integration_branch"])


def refresh_integration_branch(
    root: Path,
    task: dict[str, Any],
    base_head: str,
) -> dict[str, Any]:
    integration_branch = task["integration_branch"]
    target_before = git_commit(root, integration_branch)
    if git_is_ancestor(root, base_head, target_before):
        return {
            "base_changed": False,
            "target_before": target_before,
            "target_commit": target_before,
            "worktree_path": None,
        }

    worktree_path = find_integration_worktree(root, task)
    if worktree_path is None:
        raise OrchestrateError(
            f"integration branch {integration_branch} must be refreshed to "
            f"{base_head}, but no integration worktree exists",
            40,
        )
    current_branch = git_current_branch(worktree_path)
    if current_branch != integration_branch:
        raise OrchestrateError(
            f"integration worktree {worktree_path} is on {current_branch}, "
            f"expected {integration_branch}",
            40,
        )
    ensure_tracked_clean_worktree(worktree_path)
    result = run_git(worktree_path, "rebase", base_head, check=False)
    if result.returncode != 0:
        raise OrchestrateError(
            f"integration refresh failed: {format_git_output(result)}",
            50,
        )

    target_after = git_commit(root, integration_branch)
    if not git_is_ancestor(root, base_head, target_after):
        raise OrchestrateError(
            f"integration refresh left {integration_branch} at {target_after}, "
            f"which does not contain base {base_head}",
            50,
        )
    return {
        "base_changed": target_after != target_before,
        "target_before": target_before,
        "target_commit": target_after,
        "worktree_path": worktree_path.as_posix(),
    }


def queue_timeout_error(queue: dict[str, Any], task_id: str) -> OrchestrateError:
    entries = queue_entries(queue)
    head = entries[0]["task_id"] if entries else "<none>"
    return OrchestrateError(
        f"timed out waiting for merge queue; task {task_id} is queued behind {head}",
        20,
    )


def print_result(args: argparse.Namespace, result: dict[str, Any], text: str) -> None:
    if args.json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        print(text)


def command_init(args: argparse.Namespace) -> None:
    root = normalize_root(args.root)
    state_file = state_path(root)
    queue_file = queue_path(root)
    with workflow_locks(root, args.lock_timeout):
        if state_file.exists():
            data = load_state(root)
        else:
            data = empty_state()
            save_state(root, data)
        if queue_file.exists():
            queue = load_queue(root)
        else:
            queue = empty_queue()
            save_queue(root, queue)
    print_result(
        args,
        {
            "queue_path": queue_file.as_posix(),
            "queue": queue,
            "state_path": state_file.as_posix(),
            "state": data,
        },
        f"initialized {state_file} and {queue_file}",
    )


def command_validate(args: argparse.Namespace) -> None:
    root = normalize_root(args.root)
    data = load_state(root)
    queue = load_queue(root)
    print_result(args, {"queue": queue, "state": data}, "workflow state is valid")


def command_task_create(args: argparse.Namespace) -> None:
    validate_id(args.task_id, "task-id")
    root = normalize_root(args.root)
    identity = {
        "base_branch": args.base_branch,
        "base_commit": args.base_commit,
        "integration_branch": args.integration_branch,
    }
    expected = {
        "status": "active",
        **identity,
        "worktrees": {},
    }
    with locked_state(root, args.lock_timeout) as data:
        tasks = data["tasks"]
        current = tasks.get(args.task_id)
        if current is None:
            tasks[args.task_id] = expected
        elif any(current.get(field) != value for field, value in identity.items()):
            raise OrchestrateError(
                f"task {args.task_id} already exists with different identity",
                40,
            )
    print_result(args, {"task_id": args.task_id}, f"task {args.task_id} is active")


def command_task_status(args: argparse.Namespace) -> None:
    validate_id(args.task_id, "task-id")
    validate_status(args.status, "status")
    root = normalize_root(args.root)
    with locked_state(root, args.lock_timeout) as data:
        task = data["tasks"].get(args.task_id)
        if task is None:
            raise OrchestrateError(f"task {args.task_id} does not exist", 40)
        task["status"] = args.status
    print_result(
        args,
        {"task_id": args.task_id, "status": args.status},
        f"task {args.task_id} -> {args.status}",
    )


def command_task_show(args: argparse.Namespace) -> None:
    validate_id(args.task_id, "task-id")
    root = normalize_root(args.root)
    data = load_state(root)
    task = data["tasks"].get(args.task_id)
    if task is None:
        raise OrchestrateError(f"task {args.task_id} does not exist", 40)
    print_result(
        args,
        {"task_id": args.task_id, "task": task},
        json.dumps(task, ensure_ascii=False, indent=2),
    )


def command_task_list(args: argparse.Namespace) -> None:
    root = normalize_root(args.root)
    data = load_state(root)
    task_ids = sorted(data["tasks"])
    print_result(
        args,
        {"tasks": data["tasks"]},
        "\n".join(task_ids) if task_ids else "no tasks",
    )


def command_task_close(args: argparse.Namespace) -> None:
    validate_id(args.task_id, "task-id")
    expected_statuses = set(args.expect_status or [])
    for status in expected_statuses:
        validate_status(status, "expect-status")
    root = normalize_root(args.root)
    with locked_state(root, args.lock_timeout) as data:
        task = data["tasks"].get(args.task_id)
        if task is None:
            print_result(
                args,
                {"task_id": args.task_id, "closed": False},
                f"task {args.task_id} is already absent",
            )
            return
        if expected_statuses and task["status"] not in expected_statuses:
            raise OrchestrateError(
                f"task {args.task_id} status is {task['status']}, "
                f"expected {sorted(expected_statuses)}",
                40,
            )
        del data["tasks"][args.task_id]
    print_result(
        args, {"task_id": args.task_id, "closed": True}, f"task {args.task_id} closed"
    )


def command_lane_create(args: argparse.Namespace) -> None:
    validate_id(args.task_id, "task-id")
    validate_id(args.lane_id, "lane-id")
    validate_status(args.status, "status")
    validate_role(args.role)
    root = normalize_root(args.root)
    expected = {
        "status": args.status,
        "role": args.role,
        "branch": args.branch,
        "worktree_path": normalize_repo_path(root, args.worktree_path),
        "reports_dir": normalize_repo_path(root, args.reports_dir),
        "write_scope": normalize_scope_list(args.write_scope),
        "ignored_inputs": [parse_ignored_input(raw) for raw in args.ignored_input],
    }
    with locked_state(root, args.lock_timeout) as data:
        task = data["tasks"].get(args.task_id)
        if task is None:
            raise OrchestrateError(f"task {args.task_id} does not exist", 40)
        worktrees = task["worktrees"]
        current = worktrees.get(args.lane_id)
        if current is not None and current != expected:
            raise OrchestrateError(
                f"lane {args.task_id}/{args.lane_id} already exists with different data",
                40,
            )
        worktrees.setdefault(args.lane_id, expected)
    print_result(
        args,
        {"task_id": args.task_id, "lane_id": args.lane_id},
        f"lane {args.task_id}/{args.lane_id} is active",
    )


def command_lane_status(args: argparse.Namespace) -> None:
    validate_id(args.task_id, "task-id")
    validate_id(args.lane_id, "lane-id")
    validate_status(args.status, "status")
    root = normalize_root(args.root)
    with locked_state(root, args.lock_timeout) as data:
        lane = (
            data["tasks"].get(args.task_id, {}).get("worktrees", {}).get(args.lane_id)
        )
        if lane is None:
            raise OrchestrateError(
                f"lane {args.task_id}/{args.lane_id} does not exist", 40
            )
        lane["status"] = args.status
    print_result(
        args,
        {"task_id": args.task_id, "lane_id": args.lane_id, "status": args.status},
        f"lane {args.task_id}/{args.lane_id} -> {args.status}",
    )


def command_lane_scope_show(args: argparse.Namespace) -> None:
    validate_id(args.task_id, "task-id")
    validate_id(args.lane_id, "lane-id")
    root = normalize_root(args.root)
    data = load_state(root)
    lane = data["tasks"].get(args.task_id, {}).get("worktrees", {}).get(args.lane_id)
    if lane is None:
        raise OrchestrateError(f"lane {args.task_id}/{args.lane_id} does not exist", 40)
    write_scope = list(lane["write_scope"])
    print_result(
        args,
        {
            "task_id": args.task_id,
            "lane_id": args.lane_id,
            "write_scope": write_scope,
        },
        "\n".join(write_scope)
        if write_scope
        else f"lane {args.task_id}/{args.lane_id} has empty write scope",
    )


def command_lane_scope_update(args: argparse.Namespace) -> None:
    validate_id(args.task_id, "task-id")
    validate_id(args.lane_id, "lane-id")
    set_scope = normalize_scope_list(args.set_scope)
    add_scope = normalize_scope_list(args.add_scope)
    remove_scope = normalize_scope_list(args.remove_scope)
    expect_current = (
        normalize_scope_list(args.expect_current)
        if args.expect_current is not None
        else None
    )

    if set_scope and (add_scope or remove_scope):
        raise OrchestrateError("--set cannot be combined with --add or --remove", 40)
    if not set_scope and not add_scope and not remove_scope:
        raise OrchestrateError("provide --set, --add, or --remove", 40)
    scope_overlap = sorted(set(add_scope) & set(remove_scope))
    if scope_overlap:
        raise OrchestrateError(
            "--add and --remove cannot contain the same scope: "
            + ", ".join(scope_overlap),
            40,
        )

    root = normalize_root(args.root)
    with locked_state(root, args.lock_timeout) as data:
        task = data["tasks"].get(args.task_id)
        if task is None:
            raise OrchestrateError(f"task {args.task_id} does not exist", 40)
        if task["status"] == "merge_preview":
            raise OrchestrateError(
                f"task {args.task_id} is in merge_preview; abort preview before "
                "updating lane scope",
                40,
            )
        lane = task["worktrees"].get(args.lane_id)
        if lane is None:
            raise OrchestrateError(
                f"lane {args.task_id}/{args.lane_id} does not exist", 40
            )

        old_scope = list(lane["write_scope"])
        if expect_current is not None and old_scope != expect_current:
            raise OrchestrateError(
                f"lane {args.task_id}/{args.lane_id} write_scope changed; "
                f"expected {expect_current}, found {old_scope}",
                40,
            )

        if set_scope:
            new_scope = set_scope
        else:
            removed = set(remove_scope)
            new_scope = [scope for scope in old_scope if scope not in removed]
            for scope in add_scope:
                if scope not in new_scope:
                    new_scope.append(scope)

        if not new_scope and not args.allow_empty:
            raise OrchestrateError(
                "lane write_scope would be empty; pass --allow-empty to confirm",
                40,
            )

        lane["write_scope"] = new_scope
        changed = new_scope != old_scope

    print_result(
        args,
        {
            "task_id": args.task_id,
            "lane_id": args.lane_id,
            "old_write_scope": old_scope,
            "write_scope": new_scope,
            "changed": changed,
            "reason": args.reason,
        },
        f"lane {args.task_id}/{args.lane_id} write_scope "
        + ("updated" if changed else "unchanged"),
    )


def command_lane_remove(args: argparse.Namespace) -> None:
    validate_id(args.task_id, "task-id")
    validate_id(args.lane_id, "lane-id")
    root = normalize_root(args.root)
    removed = False
    with locked_state(root, args.lock_timeout) as data:
        task = data["tasks"].get(args.task_id)
        if task is not None:
            removed = task["worktrees"].pop(args.lane_id, None) is not None
    print_result(
        args,
        {"task_id": args.task_id, "lane_id": args.lane_id, "removed": removed},
        f"lane {args.task_id}/{args.lane_id} removed"
        if removed
        else f"lane {args.task_id}/{args.lane_id} is already absent",
    )


def command_report_path(args: argparse.Namespace) -> None:
    validate_id(args.task_id, "task-id")
    validate_id(args.lane_id, "lane-id")
    validate_id(args.agent_id, "agent-id")
    root = normalize_root(args.root)
    data = load_state(root)
    lane = data["tasks"].get(args.task_id, {}).get("worktrees", {}).get(args.lane_id)
    if lane is None:
        raise OrchestrateError(f"lane {args.task_id}/{args.lane_id} does not exist", 40)
    report_path = root / lane["reports_dir"] / f"{args.agent_id}.md"
    if args.mkdir:
        report_path.parent.mkdir(parents=True, exist_ok=True)
    print_result(args, {"report_path": report_path.as_posix()}, report_path.as_posix())


def command_queue_list(args: argparse.Namespace) -> None:
    root = normalize_root(args.root)
    queue = load_queue(root)
    entries = queue_entries(queue)
    text = "\n".join(
        f"{entry['task_id']} {entry['action']} {entry['status']}" for entry in entries
    )
    print_result(args, {"queue": queue}, text or "queue is empty")


def command_queue_status(args: argparse.Namespace) -> None:
    root = normalize_root(args.root)
    queue = load_queue(root)
    if args.task_id is None:
        entries = queue_entries(queue)
        entry = entries[0] if entries else None
    else:
        validate_id(args.task_id, "task-id")
        _, entry = find_queue_entry(queue, args.task_id)
    if entry is None:
        print_result(args, {"entry": None}, "not queued")
        return
    print_result(
        args,
        {"entry": entry},
        f"{entry['task_id']} {entry['action']} {entry['status']}",
    )


def command_queue_block(args: argparse.Namespace) -> None:
    validate_id(args.task_id, "task-id")
    root = normalize_root(args.root)
    with workflow_locks(root, args.lock_timeout):
        state = load_state(root)
        queue = load_queue(root)
        _, entry = find_queue_entry(queue, args.task_id)
        if entry is None:
            raise OrchestrateError(f"task {args.task_id} is not queued", 40)
        block_entry(entry, args.note, kind="manual")
        task = state["tasks"].get(args.task_id)
        if task is not None:
            task["status"] = "blocked"
        save_workflow(root, state, queue)
    print_result(args, {"entry": entry}, f"task {args.task_id} blocked")


def command_preview_start(args: argparse.Namespace) -> None:
    validate_id(args.task_id, "task-id")
    root = normalize_root(args.root)
    with workflow_locks(root, args.lock_timeout):
        state = load_state(root)
        queue = load_queue(root)
        task = require_task(state, args.task_id)
        ensure_on_base_branch(root, task)
        target_commit = git_commit(root, task["integration_branch"])
        base_head = git_commit(root, "HEAD")
        ensure_no_merge_in_progress(root)
        ensure_tracked_clean_worktree(root)
        ensure_untracked_files_do_not_overlap_merge_target(root, target_commit)

        index, current = find_queue_entry(queue, args.task_id)
        if current is not None and current["status"] == "merging":
            raise OrchestrateError(
                f"task {args.task_id} already holds the merge queue",
                40,
            )
        if current is not None and current["status"] == "blocked":
            raise OrchestrateError(
                f"task {args.task_id} queue entry is blocked: {current.get('note', '')}",
                40,
            )
        entry = build_queue_entry(
            task_id=args.task_id,
            task=task,
            action="preview",
            requested_by=args.requested_by,
            target_commit=target_commit,
            base_head=base_head,
            main_worktree=root,
            note=args.note,
            current=current,
        )
        index, entry = upsert_queue_entry(queue, entry)
        if index != 0:
            save_workflow(root, state, queue)
            raise queue_wait_error(queue, args.task_id)

        claim_entry(
            entry,
            action="preview",
            requested_by=args.requested_by,
            target_commit=target_commit,
            base_head=base_head,
            main_worktree=root,
            note=args.note,
        )
        task["status"] = "merge_preview"
        save_workflow(root, state, queue)

        result = run_git(
            root,
            "merge",
            "--no-overwrite-ignore",
            "--no-commit",
            "--no-ff",
            target_commit,
            check=False,
        )
        if result.returncode != 0:
            note = f"preview merge failed: {format_git_output(result)}"
            block_entry(entry, note, kind="preview_merge_failed")
            task["status"] = "blocked"
            save_workflow(root, state, queue)
            raise OrchestrateError(note, 50)

        current_target = git_merge_head(root)
        if current_target != target_commit:
            note = f"preview MERGE_HEAD is {current_target}, expected {target_commit}"
            block_entry(entry, note, kind="preview_postcondition_failed")
            task["status"] = "blocked"
            save_workflow(root, state, queue)
            raise OrchestrateError(note, 50)
        save_workflow(root, state, queue)

    print_result(
        args,
        {"entry": entry},
        f"preview started for {args.task_id} at {target_commit}",
    )


def command_preview_abort(args: argparse.Namespace) -> None:
    validate_id(args.task_id, "task-id")
    root = normalize_root(args.root)
    with workflow_locks(root, args.lock_timeout):
        state = load_state(root)
        queue = load_queue(root)
        task = require_task(state, args.task_id)
        entries = queue_entries(queue)
        index, entry = find_queue_entry(queue, args.task_id)
        if index != 0 or entry is None:
            raise OrchestrateError(f"task {args.task_id} does not hold queue head", 40)
        if entry["action"] != "preview":
            raise OrchestrateError(
                f"task {args.task_id} queue action is {entry['action']}, expected preview",
                40,
            )
        try:
            abort_preview_merge(root, entry)
        except OrchestrateError as exc:
            block_entry(entry, str(exc), kind="preview_abort_failed")
            task["status"] = "blocked"
            save_workflow(root, state, queue)
            raise
        entries.pop(0)
        task["status"] = "reviewing"
        save_workflow(root, state, queue)
    print_result(args, {"task_id": args.task_id}, f"preview aborted for {args.task_id}")


def command_final_fast_forward(args: argparse.Namespace) -> None:
    validate_id(args.task_id, "task-id")
    root = normalize_root(args.root)
    with workflow_locks(root, args.lock_timeout):
        state = load_state(root)
        queue = load_queue(root)
        task = require_task(state, args.task_id)
        ensure_on_base_branch(root, task)
        target_commit = git_commit(root, task["integration_branch"])
        entries = queue_entries(queue)
        index, entry = find_queue_entry(queue, args.task_id)

        if entry is None:
            entry = build_queue_entry(
                task_id=args.task_id,
                task=task,
                action="final",
                requested_by=args.requested_by,
                target_commit=target_commit,
                base_head=git_commit(root, "HEAD"),
                main_worktree=root,
                note=args.note,
            )
            entries.append(entry)
            index = len(entries) - 1
        if index != 0:
            save_workflow(root, state, queue)
            raise queue_wait_error(queue, args.task_id)
        if entry["status"] == "blocked":
            raise OrchestrateError(
                f"task {args.task_id} queue entry is blocked: {entry.get('note', '')}",
                40,
            )
        if entry["action"] == "preview":
            if entry.get("target_commit") != target_commit:
                note = (
                    f"preview target {entry.get('target_commit')} is stale; "
                    f"integration branch is {target_commit}"
                )
                block_entry(entry, note, kind="preview_target_stale")
                task["status"] = "blocked"
                save_workflow(root, state, queue)
                raise OrchestrateError(note, 40)
            try:
                abort_preview_merge(root, entry)
            except OrchestrateError as exc:
                block_entry(entry, str(exc), kind="preview_abort_failed")
                task["status"] = "blocked"
                save_workflow(root, state, queue)
                raise
            entry.update(
                build_queue_entry(
                    task_id=args.task_id,
                    task=task,
                    action="final",
                    requested_by=args.requested_by,
                    target_commit=target_commit,
                    base_head=git_commit(root, "HEAD"),
                    main_worktree=root,
                    note=args.note,
                    current=entry,
                )
            )
        elif entry["action"] != "final":
            raise OrchestrateError(
                f"task {args.task_id} queue action is {entry['action']}, expected final",
                40,
            )

        ensure_no_merge_in_progress(root)
        ensure_tracked_clean_worktree(root)
        ensure_untracked_files_do_not_overlap_merge_target(root, target_commit)
        base_head = git_commit(root, "HEAD")
        claim_entry(
            entry,
            action="final",
            requested_by=args.requested_by,
            target_commit=target_commit,
            base_head=base_head,
            main_worktree=root,
            note=args.note,
        )
        task["status"] = "merge_preview"
        save_workflow(root, state, queue)

        result = run_git(
            root,
            "merge",
            "--no-overwrite-ignore",
            "--ff-only",
            target_commit,
            check=False,
        )
        if result.returncode != 0:
            note = f"final fast-forward failed: {format_git_output(result)}"
            block_entry(entry, note, kind="final_fast_forward_failed")
            task["status"] = "blocked"
            save_workflow(root, state, queue)
            raise OrchestrateError(note, 50)

        head = git_commit(root, "HEAD")
        if head != target_commit:
            note = f"final fast-forward left HEAD at {head}, expected {target_commit}"
            block_entry(entry, note, kind="final_postcondition_failed")
            task["status"] = "blocked"
            save_workflow(root, state, queue)
            raise OrchestrateError(note, 50)

        entries.pop(0)
        del state["tasks"][args.task_id]
        save_workflow(root, state, queue)
    print_result(
        args,
        {"target_commit": target_commit, "task_id": args.task_id},
        f"final fast-forward completed for {args.task_id} at {target_commit}",
    )


def command_merge_retry_refresh(args: argparse.Namespace) -> None:
    validate_id(args.task_id, "task-id")
    validate_action(args.action)
    root = normalize_root(args.root)

    with workflow_locks(root, args.lock_timeout):
        state = load_state(root)
        queue = load_queue(root)
        task = require_task(state, args.task_id)
        if task["status"] != "blocked":
            raise OrchestrateError(
                f"task {args.task_id} status is {task['status']}, expected blocked",
                40,
            )

        index, entry = find_queue_entry(queue, args.task_id)
        if entry is None:
            raise OrchestrateError(f"task {args.task_id} is not queued", 40)
        if index != 0:
            raise OrchestrateError(
                f"task {args.task_id} does not hold queue head",
                40,
            )
        if entry["status"] != "blocked":
            raise OrchestrateError(
                f"task {args.task_id} queue status is {entry['status']}, expected blocked",
                40,
            )

        blocked_kind = entry.get("blocked_kind")
        if blocked_kind is None:
            raise OrchestrateError(
                f"task {args.task_id} blocked provenance is unknown; "
                "legacy blocked entries cannot use merge retry-refresh",
                40,
            )
        if blocked_kind != "integration_refresh_failed":
            raise OrchestrateError(
                f"task {args.task_id} blocked_kind is {blocked_kind}, expected "
                "integration_refresh_failed",
                40,
            )
        if entry["action"] != args.action:
            raise OrchestrateError(
                f"task {args.task_id} queue action is {entry['action']}, "
                f"expected {args.action}",
                40,
            )
        if entry["requested_by"] != args.requested_by:
            raise OrchestrateError(
                f"task {args.task_id} was requested by {entry['requested_by']}, "
                f"not {args.requested_by}",
                40,
            )
        if entry["branch"] != task["integration_branch"]:
            raise OrchestrateError(
                f"queue branch is {entry['branch']}, expected "
                f"{task['integration_branch']}",
                40,
            )
        if entry["base_branch"] != task["base_branch"]:
            raise OrchestrateError(
                f"queue base branch is {entry['base_branch']}, expected "
                f"{task['base_branch']}",
                40,
            )
        recorded_main = entry.get("main_worktree")
        if not isinstance(recorded_main, str) or not recorded_main:
            raise OrchestrateError("queue entry is missing main_worktree", 40)
        recorded_main_path = Path(recorded_main).expanduser()
        if not recorded_main_path.is_absolute():
            recorded_main_path = root / recorded_main_path
        if recorded_main_path.resolve() != root:
            raise OrchestrateError(
                f"queue main_worktree resolves to {recorded_main_path.resolve()}, "
                f"expected {root}",
                40,
            )

        ensure_on_base_branch(root, task)
        ensure_no_merge_in_progress(root)
        ensure_tracked_clean_worktree(root)

        integration_worktree = find_integration_worktree(root, task)
        if integration_worktree is None:
            raise OrchestrateError(
                f"integration worktree for {task['integration_branch']} does not exist",
                40,
            )
        current_branch = git_current_branch(integration_worktree)
        if current_branch != task["integration_branch"]:
            raise OrchestrateError(
                f"integration worktree {integration_worktree} is on {current_branch}, "
                f"expected {task['integration_branch']}",
                40,
            )
        ensure_no_merge_in_progress(integration_worktree)
        ensure_tracked_clean_worktree(integration_worktree)
        ensure_no_rebase_in_progress(integration_worktree)

        base_head, target_commit = read_merge_ref_snapshot(root, task)
        if not OBJECT_ID_PATTERN.fullmatch(args.expect_target):
            raise OrchestrateError(
                "--expect-target must be a 7-40 character hexadecimal object id "
                "or abbreviation",
                40,
            )
        expected = run_git(
            root,
            "rev-parse",
            "--verify",
            f"{args.expect_target}^{{commit}}",
            check=False,
        )
        if expected.returncode != 0:
            raise OrchestrateError(
                f"--expect-target is not a commit: {args.expect_target}",
                40,
            )
        expected_target = expected.stdout.strip()
        if expected_target != target_commit:
            raise OrchestrateError(
                f"expected target {expected_target}, current integration target is "
                f"{target_commit}",
                40,
            )
        if not git_is_ancestor(root, base_head, target_commit):
            raise OrchestrateError(
                f"integration target {target_commit} does not contain base {base_head}",
                40,
            )

        confirmed_base, confirmed_target = read_merge_ref_snapshot(root, task)
        if confirmed_base != base_head or confirmed_target != target_commit:
            raise OrchestrateError(
                "base or integration target changed while retry-refresh was checking "
                "preconditions",
                40,
            )

        note = (
            f"refresh resolved at {target_commit}; validate this target, then rerun "
            f"merge run --action {args.action}"
        )
        entry.update(
            build_queue_entry(
                task_id=args.task_id,
                task=task,
                action=args.action,
                requested_by=args.requested_by,
                target_commit=target_commit,
                base_head=base_head,
                main_worktree=root,
                note=note,
                current=entry,
            )
        )
        task["status"] = "reviewing"
        save_workflow(root, state, queue)

    print_result(
        args,
        {
            "action": args.action,
            "base_head": base_head,
            "entry": entry,
            "target_commit": target_commit,
            "task_id": args.task_id,
        },
        f"refresh retry requeued {args.task_id} at {target_commit}; validate this "
        f"target, then rerun merge run --action {args.action}",
    )


def command_merge_retry_final(args: argparse.Namespace) -> None:
    validate_id(args.task_id, "task-id")
    root = normalize_root(args.root)

    with workflow_locks(root, args.lock_timeout):
        state = load_state(root)
        queue = load_queue(root)
        task = require_task(state, args.task_id)
        if task["status"] != "blocked":
            raise OrchestrateError(
                f"task {args.task_id} status is {task['status']}, expected blocked",
                40,
            )

        entries = queue_entries(queue)
        index, entry = find_queue_entry(queue, args.task_id)
        if entry is None:
            raise OrchestrateError(f"task {args.task_id} is not queued", 40)
        if index != 0:
            raise OrchestrateError(
                f"task {args.task_id} does not hold queue head",
                40,
            )
        if entry["status"] != "blocked":
            raise OrchestrateError(
                f"task {args.task_id} queue status is {entry['status']}, expected blocked",
                40,
            )

        blocked_kind = entry.get("blocked_kind")
        if blocked_kind is None:
            raise OrchestrateError(
                f"task {args.task_id} blocked provenance is unknown; "
                "legacy blocked entries cannot use merge retry-final",
                40,
            )
        if blocked_kind != "final_fast_forward_failed":
            raise OrchestrateError(
                f"task {args.task_id} blocked_kind is {blocked_kind}, expected "
                "final_fast_forward_failed",
                40,
            )
        if entry["action"] != "final":
            raise OrchestrateError(
                f"task {args.task_id} queue action is {entry['action']}, expected final",
                40,
            )
        if entry["requested_by"] != args.requested_by:
            raise OrchestrateError(
                f"task {args.task_id} was requested by {entry['requested_by']}, "
                f"not {args.requested_by}",
                40,
            )
        if entry["branch"] != task["integration_branch"]:
            raise OrchestrateError(
                f"queue branch is {entry['branch']}, expected "
                f"{task['integration_branch']}",
                40,
            )
        if entry["base_branch"] != task["base_branch"]:
            raise OrchestrateError(
                f"queue base branch is {entry['base_branch']}, expected "
                f"{task['base_branch']}",
                40,
            )
        recorded_main = entry.get("main_worktree")
        if not isinstance(recorded_main, str) or not recorded_main:
            raise OrchestrateError("queue entry is missing main_worktree", 40)
        recorded_main_path = Path(recorded_main).expanduser()
        if not recorded_main_path.is_absolute():
            recorded_main_path = root / recorded_main_path
        if recorded_main_path.resolve() != root:
            raise OrchestrateError(
                f"queue main_worktree resolves to {recorded_main_path.resolve()}, "
                f"expected {root}",
                40,
            )

        ensure_on_base_branch(root, task)
        ensure_no_merge_in_progress(root)
        ensure_tracked_clean_worktree(root)
        base_head = git_commit(root, "HEAD")
        recorded_base = entry.get("base_head")
        if not isinstance(recorded_base, str) or not recorded_base:
            raise OrchestrateError("queue entry is missing base_head", 40)
        if recorded_base != base_head:
            raise OrchestrateError(
                f"recorded base head is {recorded_base}, current HEAD is {base_head}",
                40,
            )

        if not OBJECT_ID_PATTERN.fullmatch(args.expect_target):
            raise OrchestrateError(
                "--expect-target must be a 7-40 character hexadecimal object id "
                "or abbreviation",
                40,
            )
        expected = run_git(
            root,
            "rev-parse",
            "--verify",
            f"{args.expect_target}^{{commit}}",
            check=False,
        )
        if expected.returncode != 0:
            raise OrchestrateError(
                f"--expect-target is not a commit: {args.expect_target}",
                40,
            )
        expected_target = expected.stdout.strip()
        recorded_target = require_pinned_target(entry)
        if expected_target != recorded_target:
            raise OrchestrateError(
                f"expected target {expected_target}, recorded target is "
                f"{recorded_target}",
                40,
            )
        target_commit = git_commit(root, task["integration_branch"])
        if target_commit != recorded_target:
            raise OrchestrateError(
                f"recorded target {recorded_target}, current integration target is "
                f"{target_commit}",
                40,
            )
        ensure_untracked_files_do_not_overlap_merge_target(root, target_commit)
        if not git_is_ancestor(root, base_head, target_commit):
            raise OrchestrateError(
                f"integration target {target_commit} cannot fast-forward base "
                f"{base_head}",
                40,
            )

        confirmed_base, confirmed_target = read_merge_ref_snapshot(root, task)
        if confirmed_base != base_head or confirmed_target != target_commit:
            raise OrchestrateError(
                "base or integration target changed while retry-final was checking "
                "preconditions",
                40,
            )

        result = run_git(
            root,
            "merge",
            "--no-overwrite-ignore",
            "--ff-only",
            target_commit,
            check=False,
        )
        if result.returncode != 0:
            note = f"retry final fast-forward failed: {format_git_output(result)}"
            block_entry(entry, note, kind="final_fast_forward_failed")
            task["status"] = "blocked"
            save_workflow(root, state, queue)
            raise OrchestrateError(note, 50)

        try:
            head = git_commit(root, "HEAD")
            if head != target_commit:
                raise OrchestrateError(
                    f"HEAD is {head}, expected {target_commit}",
                    50,
                )
            ensure_no_merge_in_progress(root)
            ensure_tracked_clean_worktree(root)
        except OrchestrateError as exc:
            note = f"retry final fast-forward postcondition failed: {exc}"
            block_entry(entry, note, kind="final_fast_forward_failed")
            task["status"] = "blocked"
            save_workflow(root, state, queue)
            raise OrchestrateError(note, 50) from exc

        entries.pop(0)
        del state["tasks"][args.task_id]
        save_workflow(root, state, queue)

    print_result(
        args,
        {"target_commit": target_commit, "task_id": args.task_id},
        f"final fast-forward retry completed for {args.task_id} at {target_commit}",
    )


def command_merge_run(args: argparse.Namespace) -> None:
    validate_id(args.task_id, "task-id")
    validate_action(args.action)
    if args.timeout < 0:
        raise OrchestrateError("--timeout must be non-negative", 40)
    if args.poll_interval <= 0:
        raise OrchestrateError("--poll-interval must be positive", 40)

    root = normalize_root(args.root)
    deadline = time.monotonic() + args.timeout
    while True:
        with workflow_locks(root, args.lock_timeout):
            state = load_state(root)
            queue = load_queue(root)
            task = require_task(state, args.task_id)
            ensure_on_base_branch(root, task)
            target_commit = git_commit(root, task["integration_branch"])
            base_head = git_commit(root, "HEAD")
            entries = queue_entries(queue)
            index, current = find_queue_entry(queue, args.task_id)

            if current is not None and current["status"] == "blocked":
                raise OrchestrateError(
                    f"task {args.task_id} queue entry is blocked: "
                    f"{current.get('note', '')}",
                    40,
                )

            if current is not None and current["status"] == "merging":
                if not (
                    index == 0
                    and current["action"] == "preview"
                    and args.action == "final"
                ):
                    raise OrchestrateError(
                        f"task {args.task_id} already holds the merge queue",
                        40,
                    )
                entry = current
            elif current is not None:
                if current["action"] != args.action:
                    raise OrchestrateError(
                        f"task {args.task_id} is already queued for "
                        f"{current['action']}, requested {args.action}",
                        40,
                    )
                require_pinned_target(current)
                entry = current
            else:
                entry = build_queue_entry(
                    task_id=args.task_id,
                    task=task,
                    action=args.action,
                    requested_by=args.requested_by,
                    target_commit=target_commit,
                    base_head=base_head,
                    main_worktree=root,
                    note=args.note,
                    current=current,
                )
                index, entry = upsert_queue_entry(queue, entry)

            if index == 0:
                _run_queue_head_merge(args, root, state, queue, task, entry)
                return

            if entries and entries[0]["status"] == "blocked":
                save_workflow(root, state, queue)
                raise blocked_queue_head_error(entries[0])

            save_workflow(root, state, queue)

        if not args.wait:
            raise queue_wait_error(queue, args.task_id)
        if time.monotonic() >= deadline:
            raise queue_timeout_error(queue, args.task_id)
        time.sleep(min(args.poll_interval, max(0.0, deadline - time.monotonic())))


def _run_queue_head_merge(
    args: argparse.Namespace,
    root: Path,
    state: dict[str, Any],
    queue: dict[str, Any],
    task: dict[str, Any],
    entry: dict[str, Any],
) -> None:
    entries = queue_entries(queue)
    if not entries or entries[0] is not entry:
        raise OrchestrateError(f"task {args.task_id} does not hold queue head", 40)

    base_head = git_commit(root, "HEAD")
    preview_was_open = entry["status"] == "merging" and entry["action"] == "preview"
    if preview_was_open:
        if args.action != "final":
            raise OrchestrateError(
                f"preview is already open for {args.task_id}",
                40,
            )
        try:
            abort_preview_merge(root, entry)
        except OrchestrateError as exc:
            block_entry(entry, str(exc), kind="preview_abort_failed")
            task["status"] = "blocked"
            save_workflow(root, state, queue)
            raise
    else:
        ensure_no_merge_in_progress(root)
        ensure_tracked_clean_worktree(root)

    pinned_target = require_pinned_target(entry)
    base_head = git_commit(root, "HEAD")
    try:
        refresh = refresh_integration_branch(root, task, base_head)
        target_commit = str(refresh["target_commit"])
    except OrchestrateError as exc:
        block_entry(entry, str(exc), kind="integration_refresh_failed")
        task["status"] = "blocked"
        save_workflow(root, state, queue)
        raise

    if refresh["target_before"] != pinned_target:
        note = (
            "integration branch changed after queue entry was pinned; rerun "
            f"validation, then rerun merge run --action {args.action}"
        )
        requeue_for_validation(
            args=args,
            root=root,
            state=state,
            queue=queue,
            task=task,
            entry=entry,
            target_commit=target_commit,
            base_head=base_head,
            note=note,
            message=(
                f"integration branch changed from queued target {pinned_target} "
                f"to {refresh['target_before']}; rerun validation, then rerun "
                f"merge run --action {args.action}"
            ),
        )

    if args.action == "final" and refresh["base_changed"]:
        requeue_for_validation(
            args=args,
            root=root,
            state=state,
            queue=queue,
            task=task,
            entry=entry,
            target_commit=target_commit,
            base_head=base_head,
            note=(
                "integration branch refreshed; rerun validation, then rerun "
                "merge run --action final"
            ),
            message=(
                f"integration branch refreshed from {refresh['target_before']} to "
                f"{target_commit}; rerun validation, then rerun merge run "
                "--action final"
            ),
        )

    try:
        ensure_untracked_files_do_not_overlap_merge_target(root, target_commit)
    except OrchestrateError as exc:
        block_entry(entry, str(exc), kind="merge_target_preflight_failed")
        task["status"] = "blocked"
        save_workflow(root, state, queue)
        raise

    claim_entry(
        entry,
        action=args.action,
        requested_by=args.requested_by,
        target_commit=target_commit,
        base_head=base_head,
        main_worktree=root,
        note=args.note,
    )
    task["status"] = "merge_preview"
    save_workflow(root, state, queue)

    if args.action == "preview":
        _open_queue_managed_preview(args, root, state, queue, task, entry, refresh)
        return
    _complete_queue_managed_final(args, root, state, queue, entry, refresh)


def _open_queue_managed_preview(
    args: argparse.Namespace,
    root: Path,
    state: dict[str, Any],
    queue: dict[str, Any],
    task: dict[str, Any],
    entry: dict[str, Any],
    refresh: dict[str, Any],
) -> None:
    target_commit = str(entry["target_commit"])
    result = run_git(
        root,
        "merge",
        "--no-overwrite-ignore",
        "--no-commit",
        "--no-ff",
        target_commit,
        check=False,
    )
    if result.returncode != 0:
        note = f"preview merge failed: {format_git_output(result)}"
        block_entry(entry, note, kind="preview_merge_failed")
        task["status"] = "blocked"
        save_workflow(root, state, queue)
        raise OrchestrateError(note, 50)

    current_target = git_merge_head(root)
    if current_target != target_commit:
        note = f"preview MERGE_HEAD is {current_target}, expected {target_commit}"
        block_entry(entry, note, kind="preview_postcondition_failed")
        task["status"] = "blocked"
        save_workflow(root, state, queue)
        raise OrchestrateError(note, 50)
    save_workflow(root, state, queue)
    print_result(
        args,
        {
            "action": "preview",
            "base_changed": refresh["base_changed"],
            "entry": entry,
            "target_commit": target_commit,
            "task_id": args.task_id,
        },
        f"preview started for {args.task_id} at {target_commit}",
    )


def _complete_queue_managed_final(
    args: argparse.Namespace,
    root: Path,
    state: dict[str, Any],
    queue: dict[str, Any],
    entry: dict[str, Any],
    refresh: dict[str, Any],
) -> None:
    target_commit = str(entry["target_commit"])
    result = run_git(
        root,
        "merge",
        "--no-overwrite-ignore",
        "--ff-only",
        target_commit,
        check=False,
    )
    if result.returncode != 0:
        note = f"final fast-forward failed: {format_git_output(result)}"
        block_entry(entry, note, kind="final_fast_forward_failed")
        state["tasks"][args.task_id]["status"] = "blocked"
        save_workflow(root, state, queue)
        raise OrchestrateError(note, 50)

    head = git_commit(root, "HEAD")
    if head != target_commit:
        note = f"final fast-forward left HEAD at {head}, expected {target_commit}"
        block_entry(entry, note, kind="final_postcondition_failed")
        state["tasks"][args.task_id]["status"] = "blocked"
        save_workflow(root, state, queue)
        raise OrchestrateError(note, 50)

    queue_entries(queue).pop(0)
    del state["tasks"][args.task_id]
    save_workflow(root, state, queue)
    print_result(
        args,
        {
            "action": "final",
            "base_changed": refresh["base_changed"],
            "target_commit": target_commit,
            "task_id": args.task_id,
        },
        f"final merge completed for {args.task_id} at {target_commit}",
    )


def default_worktree_id(task_id: str, lane_id: str) -> str:
    if lane_id == "main":
        return task_id
    return f"{task_id}--{lane_id}"


def command_worktree_create_lane(args: argparse.Namespace) -> None:
    validate_id(args.task_id, "task-id")
    validate_id(args.lane_id, "lane-id")
    validate_status(args.status, "status")
    validate_role(args.role)
    root = normalize_root(args.root)
    base_commit = git_commit(root, args.base_ref)
    integration_branch = args.integration_branch or f"agent/{args.task_id}"
    branch = args.branch or (
        integration_branch
        if args.role == "integration" or args.lane_id == "main"
        else f"agent/{args.task_id}--{args.lane_id}"
    )
    worktree_id = default_worktree_id(args.task_id, args.lane_id)
    worktree_path = root / normalize_repo_path(
        root,
        args.worktree_path or f".agent_state/worktrees/trees/{worktree_id}",
    )
    reports_dir = root / normalize_repo_path(
        root,
        args.reports_dir
        or f".agent_state/worktrees/reports/{args.task_id}/{args.lane_id}",
    )
    expected_task = {
        "base_branch": args.base_branch,
        "base_commit": base_commit,
        "integration_branch": integration_branch,
    }
    expected_lane = {
        "status": args.status,
        "role": args.role,
        "branch": branch,
        "worktree_path": normalize_repo_path(root, worktree_path.as_posix()),
        "reports_dir": normalize_repo_path(root, reports_dir.as_posix()),
        "write_scope": normalize_scope_list(args.write_scope),
        "ignored_inputs": [parse_ignored_input(raw) for raw in args.ignored_input],
    }

    with file_lock(state_path(root), args.lock_timeout):
        state = load_state(root)
        task = state["tasks"].get(args.task_id)
        lane_exists = False
        if task is not None:
            if any(task.get(field) != value for field, value in expected_task.items()):
                raise OrchestrateError(
                    f"task {args.task_id} already exists with different identity",
                    40,
                )
            current_lane = task["worktrees"].get(args.lane_id)
            if current_lane is not None and current_lane != expected_lane:
                raise OrchestrateError(
                    f"lane {args.task_id}/{args.lane_id} already exists with different data",
                    40,
                )
            lane_exists = current_lane is not None

    branch_exists = git_branch_exists(root, branch)
    if branch_exists and not lane_exists:
        branch_commit = git_commit(root, branch)
        if branch_commit != base_commit:
            raise OrchestrateError(
                f"branch {branch} points to {branch_commit}, expected {base_commit}",
                40,
            )
    if not branch_exists:
        run_git(root, "branch", branch, base_commit)
    if worktree_path.exists():
        if not (worktree_path / ".git").exists():
            raise OrchestrateError(
                f"worktree path exists but is not a git worktree: {worktree_path}",
                40,
            )
        current_branch = git_current_branch(worktree_path)
        if current_branch != branch:
            raise OrchestrateError(
                f"worktree {worktree_path} is on {current_branch}, expected {branch}",
                40,
            )
    else:
        worktree_path.parent.mkdir(parents=True, exist_ok=True)
        run_git(root, "worktree", "add", worktree_path.as_posix(), branch)
    reports_dir.mkdir(parents=True, exist_ok=True)

    with locked_state(root, args.lock_timeout) as state:
        tasks = state["tasks"]
        task = tasks.get(args.task_id)
        if task is None:
            tasks[args.task_id] = {"status": "active", **expected_task, "worktrees": {}}
            task = tasks[args.task_id]
        elif any(task.get(field) != value for field, value in expected_task.items()):
            raise OrchestrateError(
                f"task {args.task_id} already exists with different identity",
                40,
            )
        current_lane = task["worktrees"].get(args.lane_id)
        if current_lane is not None and current_lane != expected_lane:
            raise OrchestrateError(
                f"lane {args.task_id}/{args.lane_id} already exists with different data",
                40,
            )
        task["worktrees"].setdefault(args.lane_id, expected_lane)

    print_result(
        args,
        {
            "branch": branch,
            "reports_dir": reports_dir.as_posix(),
            "task_id": args.task_id,
            "lane_id": args.lane_id,
            "worktree_path": worktree_path.as_posix(),
        },
        f"worktree {worktree_path} is ready on {branch}",
    )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage orchestrate workflow state")
    parser.add_argument("--root", default=".", help="main checkout root")
    parser.add_argument("--lock-timeout", type=float, default=10.0)
    subparsers = parser.add_subparsers(dest="command", required=True)

    state = subparsers.add_parser("state")
    state_subparsers = state.add_subparsers(dest="state_command", required=True)

    init = state_subparsers.add_parser("init")
    add_common_args(init)
    init.set_defaults(func=command_init)

    validate = state_subparsers.add_parser("validate")
    add_common_args(validate)
    validate.set_defaults(func=command_validate)

    task = subparsers.add_parser("task")
    task_subparsers = task.add_subparsers(dest="task_command", required=True)

    task_create = task_subparsers.add_parser("create")
    task_create.add_argument("task_id")
    task_create.add_argument("--base-branch", required=True)
    task_create.add_argument("--base-commit", required=True)
    task_create.add_argument("--integration-branch", required=True)
    add_common_args(task_create)
    task_create.set_defaults(func=command_task_create)

    task_status = task_subparsers.add_parser("status")
    task_status.add_argument("task_id")
    task_status.add_argument("--status", required=True)
    add_common_args(task_status)
    task_status.set_defaults(func=command_task_status)

    task_show = task_subparsers.add_parser("show")
    task_show.add_argument("task_id")
    add_common_args(task_show)
    task_show.set_defaults(func=command_task_show)

    task_list = task_subparsers.add_parser("list")
    add_common_args(task_list)
    task_list.set_defaults(func=command_task_list)

    task_close = task_subparsers.add_parser("close")
    task_close.add_argument("task_id")
    task_close.add_argument("--expect-status", action="append", default=[])
    add_common_args(task_close)
    task_close.set_defaults(func=command_task_close)

    lane = subparsers.add_parser("lane")
    lane_subparsers = lane.add_subparsers(dest="lane_command", required=True)

    lane_create = lane_subparsers.add_parser("create")
    lane_create.add_argument("task_id")
    lane_create.add_argument("lane_id")
    lane_create.add_argument("--role", required=True)
    lane_create.add_argument("--branch", required=True)
    lane_create.add_argument("--worktree-path", required=True)
    lane_create.add_argument("--reports-dir", required=True)
    lane_create.add_argument(
        "--write-scope", action="append", default=[], required=True
    )
    lane_create.add_argument("--ignored-input", action="append", default=[])
    lane_create.add_argument("--status", default="active")
    add_common_args(lane_create)
    lane_create.set_defaults(func=command_lane_create)

    lane_status = lane_subparsers.add_parser("status")
    lane_status.add_argument("task_id")
    lane_status.add_argument("lane_id")
    lane_status.add_argument("--status", required=True)
    add_common_args(lane_status)
    lane_status.set_defaults(func=command_lane_status)

    lane_scope = lane_subparsers.add_parser("scope")
    lane_scope_subparsers = lane_scope.add_subparsers(
        dest="lane_scope_command",
        required=True,
    )

    lane_scope_show = lane_scope_subparsers.add_parser("show")
    lane_scope_show.add_argument("task_id")
    lane_scope_show.add_argument("lane_id")
    add_common_args(lane_scope_show)
    lane_scope_show.set_defaults(func=command_lane_scope_show)

    lane_scope_update = lane_scope_subparsers.add_parser("update")
    lane_scope_update.add_argument("task_id")
    lane_scope_update.add_argument("lane_id")
    lane_scope_update.add_argument(
        "--set", action="append", default=[], dest="set_scope"
    )
    lane_scope_update.add_argument(
        "--add", action="append", default=[], dest="add_scope"
    )
    lane_scope_update.add_argument(
        "--remove",
        action="append",
        default=[],
        dest="remove_scope",
    )
    lane_scope_update.add_argument(
        "--expect-current",
        action="append",
        dest="expect_current",
    )
    lane_scope_update.add_argument("--allow-empty", action="store_true")
    lane_scope_update.add_argument("--reason", required=True)
    add_common_args(lane_scope_update)
    lane_scope_update.set_defaults(func=command_lane_scope_update)

    lane_remove = lane_subparsers.add_parser("remove")
    lane_remove.add_argument("task_id")
    lane_remove.add_argument("lane_id")
    add_common_args(lane_remove)
    lane_remove.set_defaults(func=command_lane_remove)

    report = subparsers.add_parser("report")
    report_subparsers = report.add_subparsers(dest="report_command", required=True)

    report_path = report_subparsers.add_parser("path")
    report_path.add_argument("task_id")
    report_path.add_argument("lane_id")
    report_path.add_argument("agent_id")
    report_path.add_argument("--mkdir", action="store_true")
    add_common_args(report_path)
    report_path.set_defaults(func=command_report_path)

    queue = subparsers.add_parser("queue")
    queue_subparsers = queue.add_subparsers(dest="queue_command", required=True)

    queue_list = queue_subparsers.add_parser("list")
    add_common_args(queue_list)
    queue_list.set_defaults(func=command_queue_list)

    queue_status = queue_subparsers.add_parser("status")
    queue_status.add_argument("task_id", nargs="?")
    add_common_args(queue_status)
    queue_status.set_defaults(func=command_queue_status)

    queue_block = queue_subparsers.add_parser("block")
    queue_block.add_argument("task_id")
    queue_block.add_argument("--note", required=True)
    add_common_args(queue_block)
    queue_block.set_defaults(func=command_queue_block)

    preview = subparsers.add_parser("preview")
    preview_subparsers = preview.add_subparsers(dest="preview_command", required=True)

    preview_start = preview_subparsers.add_parser("start")
    preview_start.add_argument("task_id")
    preview_start.add_argument("--requested-by", required=True)
    preview_start.add_argument("--note", default="")
    add_common_args(preview_start)
    preview_start.set_defaults(func=command_preview_start)

    preview_abort = preview_subparsers.add_parser("abort")
    preview_abort.add_argument("task_id")
    add_common_args(preview_abort)
    preview_abort.set_defaults(func=command_preview_abort)

    final = subparsers.add_parser("final")
    final_subparsers = final.add_subparsers(dest="final_command", required=True)

    final_ff = final_subparsers.add_parser("fast-forward")
    final_ff.add_argument("task_id")
    final_ff.add_argument("--requested-by", required=True)
    final_ff.add_argument("--note", default="")
    add_common_args(final_ff)
    final_ff.set_defaults(func=command_final_fast_forward)

    merge = subparsers.add_parser("merge")
    merge_subparsers = merge.add_subparsers(dest="merge_command", required=True)

    merge_run = merge_subparsers.add_parser("run")
    merge_run.add_argument("task_id")
    merge_run.add_argument("--action", required=True, choices=sorted(QUEUE_ACTIONS))
    merge_run.add_argument("--requested-by", required=True)
    merge_run.add_argument("--note", default="")
    merge_run.add_argument("--wait", action="store_true")
    merge_run.add_argument("--timeout", type=float, default=3600.0)
    merge_run.add_argument("--poll-interval", type=float, default=2.0)
    add_common_args(merge_run)
    merge_run.set_defaults(func=command_merge_run)

    merge_retry_refresh = merge_subparsers.add_parser("retry-refresh")
    merge_retry_refresh.add_argument("task_id")
    merge_retry_refresh.add_argument(
        "--action",
        required=True,
        choices=sorted(QUEUE_ACTIONS),
    )
    merge_retry_refresh.add_argument("--requested-by", required=True)
    merge_retry_refresh.add_argument("--expect-target", required=True)
    add_common_args(merge_retry_refresh)
    merge_retry_refresh.set_defaults(func=command_merge_retry_refresh)

    merge_retry_final = merge_subparsers.add_parser("retry-final")
    merge_retry_final.add_argument("task_id")
    merge_retry_final.add_argument("--requested-by", required=True)
    merge_retry_final.add_argument("--expect-target", required=True)
    add_common_args(merge_retry_final)
    merge_retry_final.set_defaults(func=command_merge_retry_final)

    worktree = subparsers.add_parser("worktree")
    worktree_subparsers = worktree.add_subparsers(
        dest="worktree_command",
        required=True,
    )

    worktree_create_lane = worktree_subparsers.add_parser("create-lane")
    worktree_create_lane.add_argument("task_id")
    worktree_create_lane.add_argument("lane_id")
    worktree_create_lane.add_argument("--base-branch", required=True)
    worktree_create_lane.add_argument("--base-ref", default="HEAD")
    worktree_create_lane.add_argument("--integration-branch")
    worktree_create_lane.add_argument("--role", default="lane")
    worktree_create_lane.add_argument("--branch")
    worktree_create_lane.add_argument("--worktree-path")
    worktree_create_lane.add_argument("--reports-dir")
    worktree_create_lane.add_argument(
        "--write-scope", action="append", default=[], required=True
    )
    worktree_create_lane.add_argument("--ignored-input", action="append", default=[])
    worktree_create_lane.add_argument("--status", default="active")
    add_common_args(worktree_create_lane)
    worktree_create_lane.set_defaults(func=command_worktree_create_lane)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except OrchestrateError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
