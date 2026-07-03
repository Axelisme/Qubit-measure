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
TASK_STATUSES = {"active", "reviewing", "merge_preview", "blocked"}
LANE_ROLES = {"lane", "integration"}
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


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
    return raw_scope.replace("\\", "/").removeprefix("./")


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
        validate_queue_status(str(entry.get("status")), 10)
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
            f"main checkout already has a merge preview for {current}",
            40,
        )


def ensure_tracked_clean_worktree(root: Path) -> None:
    status = run_git(
        root, "status", "--porcelain", "--untracked-files=no"
    ).stdout.strip()
    if status:
        raise OrchestrateError(f"main checkout has tracked changes:\n{status}", 40)


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
            "started_at": now_utc(),
            "note": note,
            "token": uuid4().hex,
            "target_commit": target_commit,
            "base_head": base_head,
            "main_worktree": main_worktree.as_posix(),
        }
    )


def block_entry(entry: dict[str, Any], note: str) -> None:
    entry["status"] = "blocked"
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
        "write_scope": [normalize_scope(scope) for scope in args.write_scope],
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
        block_entry(entry, args.note)
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
            block_entry(entry, note)
            task["status"] = "blocked"
            save_workflow(root, state, queue)
            raise OrchestrateError(note, 50)

        current_target = git_merge_head(root)
        if current_target != target_commit:
            note = f"preview MERGE_HEAD is {current_target}, expected {target_commit}"
            block_entry(entry, note)
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
            block_entry(entry, str(exc))
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
                block_entry(entry, note)
                task["status"] = "blocked"
                save_workflow(root, state, queue)
                raise OrchestrateError(note, 40)
            try:
                abort_preview_merge(root, entry)
            except OrchestrateError as exc:
                block_entry(entry, str(exc))
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
            block_entry(entry, note)
            task["status"] = "blocked"
            save_workflow(root, state, queue)
            raise OrchestrateError(note, 50)

        head = git_commit(root, "HEAD")
        if head != target_commit:
            note = f"final fast-forward left HEAD at {head}, expected {target_commit}"
            block_entry(entry, note)
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
        "write_scope": [normalize_scope(scope) for scope in args.write_scope],
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
