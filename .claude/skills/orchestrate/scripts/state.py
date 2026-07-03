#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

STATE_VERSION = 2
TASK_STATUSES = {"active", "reviewing", "merge_preview", "blocked"}
LANE_ROLES = {"lane", "integration"}
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class OrchestrateError(RuntimeError):
    def __init__(self, message: str, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def state_path(root: Path) -> Path:
    return root / ".agent_state" / "worktrees" / "state.json"


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


def print_result(args: argparse.Namespace, result: dict[str, Any], text: str) -> None:
    if args.json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        print(text)


def command_init(args: argparse.Namespace) -> None:
    root = normalize_root(args.root)
    path = state_path(root)
    with file_lock(path, args.lock_timeout):
        if path.exists():
            data = load_state(root)
        else:
            data = empty_state()
            save_state(root, data)
    print_result(args, {"path": path.as_posix(), "state": data}, f"initialized {path}")


def command_validate(args: argparse.Namespace) -> None:
    root = normalize_root(args.root)
    data = load_state(root)
    print_result(args, {"state": data}, "state is valid")


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


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage orchestrate state.json")
    parser.add_argument("--root", default=".", help="main checkout root")
    parser.add_argument("--lock-timeout", type=float, default=10.0)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init")
    add_common_args(init)
    init.set_defaults(func=command_init)

    validate = subparsers.add_parser("validate")
    add_common_args(validate)
    validate.set_defaults(func=command_validate)

    task_create = subparsers.add_parser("task-create")
    task_create.add_argument("task_id")
    task_create.add_argument("--base-branch", required=True)
    task_create.add_argument("--base-commit", required=True)
    task_create.add_argument("--integration-branch", required=True)
    add_common_args(task_create)
    task_create.set_defaults(func=command_task_create)

    task_status = subparsers.add_parser("task-status")
    task_status.add_argument("task_id")
    task_status.add_argument("--status", required=True)
    add_common_args(task_status)
    task_status.set_defaults(func=command_task_status)

    task_show = subparsers.add_parser("task-show")
    task_show.add_argument("task_id")
    add_common_args(task_show)
    task_show.set_defaults(func=command_task_show)

    task_list = subparsers.add_parser("task-list")
    add_common_args(task_list)
    task_list.set_defaults(func=command_task_list)

    task_close = subparsers.add_parser("task-close")
    task_close.add_argument("task_id")
    task_close.add_argument("--expect-status", action="append", default=[])
    add_common_args(task_close)
    task_close.set_defaults(func=command_task_close)

    lane_create = subparsers.add_parser("lane-create")
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

    lane_status = subparsers.add_parser("lane-status")
    lane_status.add_argument("task_id")
    lane_status.add_argument("lane_id")
    lane_status.add_argument("--status", required=True)
    add_common_args(lane_status)
    lane_status.set_defaults(func=command_lane_status)

    lane_remove = subparsers.add_parser("lane-remove")
    lane_remove.add_argument("task_id")
    lane_remove.add_argument("lane_id")
    add_common_args(lane_remove)
    lane_remove.set_defaults(func=command_lane_remove)

    report_path = subparsers.add_parser("report-path")
    report_path.add_argument("task_id")
    report_path.add_argument("lane_id")
    report_path.add_argument("agent_id")
    report_path.add_argument("--mkdir", action="store_true")
    add_common_args(report_path)
    report_path.set_defaults(func=command_report_path)

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
