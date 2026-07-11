#!/usr/bin/env python3
"""Manage the repo-local candidate backlog using only the Python stdlib."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

STATUSES = ("inbox", "planned", "resolved", "closed")
KINDS = (
    "defect",
    "missing-capability",
    "design-debt",
    "technical-debt",
    "test-gap",
    "documentation-gap",
    "workflow-friction",
    "observability-gap",
    "performance-opportunity",
    "product-idea",
)
RESOLUTIONS = (
    "implemented",
    "declined",
    "duplicate",
    "obsolete",
    "not-reproducible",
    "out-of-scope",
)
ID_RE = re.compile(r"^BL-\d{8}T\d{12}Z-[a-z0-9]+(?:-[a-z0-9]+)*$")
TASK_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
META_RE = re.compile(r"\A<!-- backlog-metadata\n(.+?)\n-->\n", re.DOTALL)


class BacklogError(Exception):
    pass


def now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def normalize_title(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def required_text(name: str, value: str | None) -> str:
    if value is None or not value.strip():
        raise BacklogError(f"{name} must not be empty")
    return value.strip()


def validate_task_id(value: str) -> str:
    task_id = required_text("task-id", value)
    if not TASK_ID_RE.fullmatch(task_id):
        raise BacklogError(f"invalid task-id: {task_id}")
    return task_id


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", normalized.casefold()).strip("-")
    return (slug or "item")[:60].rstrip("-")


def backlog_root(root: Path) -> Path:
    return root.resolve() / ".agent_state" / "backlog"


def ensure_dirs(base: Path) -> None:
    for status in STATUSES:
        (base / status).mkdir(parents=True, exist_ok=True)


@contextmanager
def locked(base: Path, timeout: float = 10.0) -> Iterator[None]:
    ensure_dirs(base)
    lock = base / ".lock"
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise BacklogError("candidate backlog lock timeout") from None
            time.sleep(0.05)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def atomic_write(path: Path, text: str) -> None:
    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as tmp:
        tmp.write(text)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def render(meta: dict[str, Any]) -> str:
    metadata = json.dumps(
        meta, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    sections = [
        ("Observation", meta["observation"]),
        ("Evidence", meta["evidence"]),
        ("Impact", meta["impact"]),
        ("Desired outcome", meta["desired_outcome"]),
    ]
    for key, heading in (
        ("constraints", "Constraints"),
        ("suggested_direction", "Suggested direction"),
    ):
        if meta.get(key):
            sections.append((heading, meta[key]))
    if meta.get("resolution"):
        sections.append(
            ("Resolution", meta.get("resolution_note") or meta["resolution"])
        )
    body = "\n\n".join(f"## {heading}\n\n{text}" for heading, text in sections)
    return f"<!-- backlog-metadata\n{metadata}\n-->\n\n# {meta['title']}\n\n{body}\n"


def read_item(path: Path) -> dict[str, Any]:
    match = META_RE.match(path.read_text(encoding="utf-8"))
    if match is None:
        raise BacklogError(f"invalid metadata block: {path}")
    data = json.loads(match.group(1))
    if not isinstance(data, dict) or data.get("id") != path.stem:
        raise BacklogError(f"metadata identity mismatch: {path}")
    validate_item(data, path)
    return data


def validate_item(data: dict[str, Any], path: Path) -> None:
    item_id = data.get("id")
    if not isinstance(item_id, str) or not ID_RE.fullmatch(item_id):
        raise BacklogError(f"invalid item id in metadata: {path}")
    for key in (
        "title",
        "source_task",
        "observation",
        "evidence",
        "impact",
        "desired_outcome",
    ):
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise BacklogError(f"invalid required field {key}: {path}")
    if data.get("kind") not in KINDS:
        raise BacklogError(f"invalid kind: {path}")
    if data.get("status") not in STATUSES:
        raise BacklogError(f"invalid status: {path}")
    areas = data.get("area")
    if (
        not isinstance(areas, list)
        or not areas
        or not all(isinstance(area, str) and area.strip() for area in areas)
    ):
        raise BacklogError(f"invalid area: {path}")


def all_items(base: Path) -> list[tuple[Path, dict[str, Any]]]:
    ensure_dirs(base)
    result = []
    for status in STATUSES:
        for path in sorted((base / status).glob("BL-*.md")):
            item = read_item(path)
            if item.get("status") != status:
                raise BacklogError(f"status/path mismatch: {path}")
            result.append((path, item))
    return result


def find_item(base: Path, item_id: str) -> tuple[Path, dict[str, Any]]:
    if not ID_RE.fullmatch(item_id):
        raise BacklogError(f"invalid backlog id: {item_id}")
    matches = [(path, item) for path, item in all_items(base) if item["id"] == item_id]
    if len(matches) != 1:
        raise BacklogError(f"backlog item not found: {item_id}")
    return matches[0]


def command_add(args: argparse.Namespace) -> dict[str, Any]:
    base = backlog_root(args.root)
    with locked(base):
        title = required_text("title", args.title)
        areas = sorted({required_text("area", area) for area in args.area})
        title_key = normalize_title(title)
        for _, existing in all_items(base):
            if normalize_title(existing["title"]) == title_key:
                raise BacklogError(f"duplicate title; existing item: {existing['id']}")
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        item_id = f"BL-{timestamp}-{slugify(title)}"
        meta: dict[str, Any] = {
            "id": item_id,
            "created_at": now(),
            "updated_at": now(),
            "status": "inbox",
            "kind": args.kind,
            "area": areas,
            "priority_hint": args.priority_hint,
            "source_task": required_text("source-task", args.source_task),
            "title": title,
            "observation": required_text("observation", args.observation),
            "evidence": required_text("evidence", args.evidence),
            "impact": required_text("impact", args.impact),
            "desired_outcome": required_text("desired-outcome", args.desired_outcome),
        }
        if args.constraints:
            meta["constraints"] = args.constraints.strip()
        if args.suggested_direction:
            meta["suggested_direction"] = args.suggested_direction.strip()
        path = base / "inbox" / f"{item_id}.md"
        atomic_write(path, render(meta))
        return meta


def command_list(args: argparse.Namespace) -> list[dict[str, Any]]:
    base = backlog_root(args.root)
    with locked(base):
        items = [item for _, item in all_items(base)]
        if args.status:
            items = [item for item in items if item["status"] == args.status]
        if args.kind:
            items = [item for item in items if item["kind"] == args.kind]
        if args.area:
            items = [item for item in items if args.area in item["area"]]
        return items


def move_item(base: Path, source: Path, meta: dict[str, Any], status: str) -> None:
    destination = base / status / source.name
    if destination.exists():
        raise BacklogError(f"destination already exists: {destination}")
    meta["status"] = status
    meta["updated_at"] = now()
    atomic_write(destination, render(meta))
    source.unlink()


def command_plan(args: argparse.Namespace) -> dict[str, Any]:
    base = backlog_root(args.root)
    with locked(base):
        path, meta = find_item(base, args.item_id)
        if meta["status"] != "inbox":
            raise BacklogError("only inbox items can be planned")
        task_id = validate_task_id(args.task_id)
        task_plan = (
            args.root.resolve() / ".agent_state" / "plans" / task_id / "task_plan.md"
        )
        if not task_plan.is_file():
            raise BacklogError(f"formal task plan not found: {task_plan}")
        meta["planned_task"] = task_id
        meta["planned_at"] = now()
        move_item(base, path, meta, "planned")
        return meta


def command_close(args: argparse.Namespace) -> dict[str, Any]:
    base = backlog_root(args.root)
    with locked(base):
        path, meta = find_item(base, args.item_id)
        if meta["status"] not in ("inbox", "planned"):
            raise BacklogError("only inbox or planned items can be closed")
        if args.resolution == "implemented":
            if meta["status"] != "planned":
                raise BacklogError("implemented resolution requires a planned item")
            if args.task_id != meta.get("planned_task"):
                raise BacklogError("implemented resolution requires the bound task-id")
            if not args.commit or not args.validation:
                raise BacklogError(
                    "implemented resolution requires commit and validation"
                )
            commits = [required_text("commit", value) for value in args.commit]
            validations = [
                required_text("validation", value) for value in args.validation
            ]
            meta["resolved_by_task"] = args.task_id
            meta["commits"] = commits
            meta["validation"] = validations
            destination = "resolved"
        else:
            destination = "closed"
        if args.resolution == "duplicate":
            if not args.duplicate_of or args.duplicate_of == args.item_id:
                raise BacklogError("duplicate resolution requires another canonical ID")
            find_item(base, args.duplicate_of)
            meta["duplicate_of"] = args.duplicate_of
        elif args.duplicate_of:
            raise BacklogError("--duplicate-of is only valid for duplicate resolution")
        meta["resolution"] = args.resolution
        meta["resolution_note"] = args.note or args.resolution
        meta["closed_at"] = now()
        move_item(base, path, meta, destination)
        return meta


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--root", type=Path, required=True)
    result.add_argument("--json", action="store_true")
    sub = result.add_subparsers(dest="command", required=True)
    add = sub.add_parser("add")
    add.add_argument("--kind", choices=KINDS, required=True)
    add.add_argument("--area", action="append", required=True)
    add.add_argument("--source-task", required=True)
    add.add_argument("--title", required=True)
    add.add_argument("--observation", required=True)
    add.add_argument("--evidence", required=True)
    add.add_argument("--impact", required=True)
    add.add_argument("--desired-outcome", required=True)
    add.add_argument(
        "--priority-hint", choices=("low", "medium", "high"), default="medium"
    )
    add.add_argument("--constraints")
    add.add_argument("--suggested-direction")
    listing = sub.add_parser("list")
    listing.add_argument("--status", choices=STATUSES)
    listing.add_argument("--kind", choices=KINDS)
    listing.add_argument("--area")
    listing.add_argument("--json", action="store_true")
    plan = sub.add_parser("plan")
    plan.add_argument("item_id")
    plan.add_argument("--task-id", required=True)
    close = sub.add_parser("close")
    close.add_argument("item_id")
    close.add_argument("--resolution", choices=RESOLUTIONS, required=True)
    close.add_argument("--task-id")
    close.add_argument("--commit", action="append")
    close.add_argument("--validation", action="append")
    close.add_argument("--duplicate-of")
    close.add_argument("--note")
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "add":
            payload: Any = command_add(args)
        elif args.command == "list":
            payload = command_list(args)
        elif args.command == "plan":
            payload = command_plan(args)
        else:
            payload = command_close(args)
    except (BacklogError, json.JSONDecodeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 40
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    elif isinstance(payload, list):
        for item in payload:
            print(f"{item['id']}\t{item['status']}\t{item['kind']}\t{item['title']}")
    else:
        print(payload["id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
