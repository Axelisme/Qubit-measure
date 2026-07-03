#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import state as state_store

QUEUE_VERSION = 1
QUEUE_ACTIONS = {"preview", "final"}
QUEUE_STATUSES = {"queued", "merging", "blocked"}
RELEASE_RESULTS = {"preview-aborted", "final-complete", "abandoned"}


def queue_path(root: Path) -> Path:
    return root / ".agent_state" / "worktrees" / "merge_queue.json"


def empty_queue() -> dict[str, Any]:
    return {"version": QUEUE_VERSION, "queue": []}


def now_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_action(value: str) -> None:
    if value not in QUEUE_ACTIONS:
        raise state_store.OrchestrateError(
            f"action must be one of {sorted(QUEUE_ACTIONS)}: {value!r}",
            40,
        )


def validate_queue_status(value: str) -> None:
    if value not in QUEUE_STATUSES:
        raise state_store.OrchestrateError(
            f"queue status must be one of {sorted(QUEUE_STATUSES)}: {value!r}",
            10,
        )


def load_queue(root: Path) -> dict[str, Any]:
    data = state_store.load_json(queue_path(root), empty_queue())
    validate_queue(data)
    return data


def save_queue(root: Path, data: dict[str, Any]) -> None:
    validate_queue(data)
    state_store.atomic_write_json(queue_path(root), data)


def validate_queue(data: dict[str, Any]) -> None:
    if data.get("version") != QUEUE_VERSION:
        raise state_store.OrchestrateError(
            f"merge queue version must be {QUEUE_VERSION}",
            10,
        )
    queue = data.get("queue")
    if not isinstance(queue, list):
        raise state_store.OrchestrateError("merge_queue.queue must be a list", 10)
    seen: set[str] = set()
    for index, entry in enumerate(queue):
        if not isinstance(entry, dict):
            raise state_store.OrchestrateError(
                f"queue entry {index} must be an object",
                10,
            )
        task_id = entry.get("task_id")
        if not isinstance(task_id, str):
            raise state_store.OrchestrateError(
                f"queue entry {index}.task_id must be a string",
                10,
            )
        state_store.validate_id(task_id, "task-id")
        if task_id in seen:
            raise state_store.OrchestrateError(
                f"queue contains duplicate task {task_id}",
                10,
            )
        seen.add(task_id)
        for field in ("branch", "base_branch", "requested_by", "note"):
            if not isinstance(entry.get(field), str):
                raise state_store.OrchestrateError(
                    f"queue entry {index}.{field} must be a string",
                    10,
                )
        action = entry.get("action")
        if not isinstance(action, str):
            raise state_store.OrchestrateError(
                f"queue entry {index}.action must be a string",
                10,
            )
        validate_action(action)
        status = entry.get("status")
        if not isinstance(status, str):
            raise state_store.OrchestrateError(
                f"queue entry {index}.status must be a string",
                10,
            )
        validate_queue_status(status)
        if entry.get("started_at") is not None and not isinstance(
            entry.get("started_at"),
            str,
        ):
            raise state_store.OrchestrateError(
                f"queue entry {index}.started_at must be null or a string",
                10,
            )
        if not isinstance(entry.get("enqueued_at"), str):
            raise state_store.OrchestrateError(
                f"queue entry {index}.enqueued_at must be a string",
                10,
            )


def find_entry(queue: list[dict[str, Any]], task_id: str) -> dict[str, Any] | None:
    return next((entry for entry in queue if entry["task_id"] == task_id), None)


def assert_task_branch(root: Path, task_id: str, branch: str) -> None:
    state = state_store.load_state(root)
    task = state["tasks"].get(task_id)
    if task is None:
        raise state_store.OrchestrateError(f"task {task_id} does not exist", 40)
    integration_branch = task["integration_branch"]
    if integration_branch != branch:
        raise state_store.OrchestrateError(
            f"task {task_id} integration branch is {integration_branch}, got {branch}",
            40,
        )


def update_task_status(root: Path, task_id: str, status: str, timeout: float) -> None:
    with state_store.locked_state(root, timeout) as data:
        task = data["tasks"].get(task_id)
        if task is not None:
            task["status"] = status


def print_result(args: argparse.Namespace, result: dict[str, Any], text: str) -> None:
    if args.json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        print(text)


def with_locked_queue(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    root = state_store.normalize_root(args.root)
    path = queue_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    return root, load_queue(root)


def command_init(args: argparse.Namespace) -> None:
    root = state_store.normalize_root(args.root)
    path = queue_path(root)
    with state_store.file_lock(path, args.lock_timeout):
        if path.exists():
            data = load_queue(root)
        else:
            data = empty_queue()
            save_queue(root, data)
    print_result(args, {"path": path.as_posix(), "queue": data}, f"initialized {path}")


def command_validate(args: argparse.Namespace) -> None:
    root = state_store.normalize_root(args.root)
    data = load_queue(root)
    print_result(args, {"queue": data}, "merge queue is valid")


def command_list(args: argparse.Namespace) -> None:
    root = state_store.normalize_root(args.root)
    data = load_queue(root)
    lines = [
        f"{entry['task_id']} {entry['action']} {entry['status']}"
        for entry in data["queue"]
    ]
    print_result(args, {"queue": data["queue"]}, "\n".join(lines) if lines else "empty")


def command_status(args: argparse.Namespace) -> None:
    state_store.validate_id(args.task_id, "task-id")
    root = state_store.normalize_root(args.root)
    data = load_queue(root)
    entry = find_entry(data["queue"], args.task_id)
    if entry is None:
        raise state_store.OrchestrateError(f"task {args.task_id} is not queued", 40)
    print_result(
        args, {"entry": entry}, json.dumps(entry, ensure_ascii=False, indent=2)
    )


def command_enqueue(args: argparse.Namespace) -> None:
    state_store.validate_id(args.task_id, "task-id")
    state_store.validate_id(args.requested_by, "requested-by")
    validate_action(args.action)
    root = state_store.normalize_root(args.root)
    assert_task_branch(root, args.task_id, args.branch)
    path = queue_path(root)
    with state_store.file_lock(path, args.lock_timeout):
        data = load_queue(root)
        queue = data["queue"]
        current = find_entry(queue, args.task_id)
        next_entry = {
            "task_id": args.task_id,
            "branch": args.branch,
            "base_branch": args.base_branch,
            "action": args.action,
            "status": "queued",
            "requested_by": args.requested_by,
            "enqueued_at": current["enqueued_at"] if current else now_utc(),
            "started_at": None,
            "note": args.note,
        }
        if current is None:
            queue.append(next_entry)
        elif current["status"] == "blocked":
            raise state_store.OrchestrateError(
                f"task {args.task_id} is blocked in merge queue",
                40,
            )
        elif current["status"] == "merging":
            comparable = dict(current)
            comparable["started_at"] = None
            if comparable != next_entry:
                raise state_store.OrchestrateError(
                    f"task {args.task_id} is already merging with different data",
                    40,
                )
        else:
            current.update(next_entry)
        save_queue(root, data)
    print_result(args, {"task_id": args.task_id}, f"task {args.task_id} queued")


def command_claim(args: argparse.Namespace) -> None:
    state_store.validate_id(args.task_id, "task-id")
    root = state_store.normalize_root(args.root)
    path = queue_path(root)
    with state_store.file_lock(path, args.lock_timeout):
        data = load_queue(root)
        queue = data["queue"]
        if not queue or queue[0]["task_id"] != args.task_id:
            head = queue[0]["task_id"] if queue else None
            raise state_store.OrchestrateError(
                f"task {args.task_id} is not queue head; head is {head}",
                20,
            )
        entry = queue[0]
        if entry["status"] == "blocked":
            raise state_store.OrchestrateError(
                f"task {args.task_id} is blocked in merge queue",
                40,
            )
        if entry["status"] == "queued":
            entry["status"] = "merging"
            entry["started_at"] = now_utc()
        update_task_status(root, args.task_id, "merge_preview", args.lock_timeout)
        save_queue(root, data)
    print_result(args, {"entry": entry}, f"task {args.task_id} claimed")


def command_assert_held(args: argparse.Namespace) -> None:
    state_store.validate_id(args.task_id, "task-id")
    root = state_store.normalize_root(args.root)
    data = load_queue(root)
    queue = data["queue"]
    if (
        not queue
        or queue[0]["task_id"] != args.task_id
        or queue[0]["status"] != "merging"
    ):
        head = queue[0] if queue else None
        raise state_store.OrchestrateError(
            f"task {args.task_id} does not hold merge queue head; head is {head}",
            20,
        )
    print_result(args, {"entry": queue[0]}, f"task {args.task_id} holds queue head")


def command_release(args: argparse.Namespace) -> None:
    state_store.validate_id(args.task_id, "task-id")
    if args.result not in RELEASE_RESULTS:
        raise state_store.OrchestrateError(
            f"result must be one of {sorted(RELEASE_RESULTS)}: {args.result!r}",
            40,
        )
    root = state_store.normalize_root(args.root)
    path = queue_path(root)
    removed = False
    with state_store.file_lock(path, args.lock_timeout):
        data = load_queue(root)
        queue = data["queue"]
        if queue and queue[0]["task_id"] == args.task_id:
            queue.pop(0)
            removed = True
        elif find_entry(queue, args.task_id) is not None:
            raise state_store.OrchestrateError(
                f"task {args.task_id} is queued but is not queue head",
                20,
            )
        if removed and args.result == "preview-aborted":
            update_task_status(root, args.task_id, "reviewing", args.lock_timeout)
        save_queue(root, data)
    print_result(
        args,
        {"task_id": args.task_id, "removed": removed, "result": args.result},
        f"task {args.task_id} released"
        if removed
        else f"task {args.task_id} is already absent",
    )


def command_block(args: argparse.Namespace) -> None:
    state_store.validate_id(args.task_id, "task-id")
    root = state_store.normalize_root(args.root)
    path = queue_path(root)
    with state_store.file_lock(path, args.lock_timeout):
        data = load_queue(root)
        entry = find_entry(data["queue"], args.task_id)
        if entry is None:
            raise state_store.OrchestrateError(f"task {args.task_id} is not queued", 40)
        entry["status"] = "blocked"
        entry["note"] = args.note
        update_task_status(root, args.task_id, "blocked", args.lock_timeout)
        save_queue(root, data)
    print_result(args, {"entry": entry}, f"task {args.task_id} blocked")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage orchestrate merge_queue.json")
    parser.add_argument("--root", default=".", help="main checkout root")
    parser.add_argument("--lock-timeout", type=float, default=10.0)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init")
    add_common_args(init)
    init.set_defaults(func=command_init)

    validate = subparsers.add_parser("validate")
    add_common_args(validate)
    validate.set_defaults(func=command_validate)

    list_cmd = subparsers.add_parser("list")
    add_common_args(list_cmd)
    list_cmd.set_defaults(func=command_list)

    enqueue = subparsers.add_parser("enqueue")
    enqueue.add_argument("task_id")
    enqueue.add_argument("--branch", required=True)
    enqueue.add_argument("--base-branch", required=True)
    enqueue.add_argument("--action", required=True)
    enqueue.add_argument("--requested-by", required=True)
    enqueue.add_argument("--note", default="")
    add_common_args(enqueue)
    enqueue.set_defaults(func=command_enqueue)

    claim = subparsers.add_parser("claim")
    claim.add_argument("task_id")
    add_common_args(claim)
    claim.set_defaults(func=command_claim)

    assert_held = subparsers.add_parser("assert-held")
    assert_held.add_argument("task_id")
    add_common_args(assert_held)
    assert_held.set_defaults(func=command_assert_held)

    release = subparsers.add_parser("release")
    release.add_argument("task_id")
    release.add_argument("--result", required=True)
    add_common_args(release)
    release.set_defaults(func=command_release)

    block = subparsers.add_parser("block")
    block.add_argument("task_id")
    block.add_argument("--note", required=True)
    add_common_args(block)
    block.set_defaults(func=command_block)

    status = subparsers.add_parser("status")
    status.add_argument("task_id")
    add_common_args(status)
    status.set_defaults(func=command_status)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except state_store.OrchestrateError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
