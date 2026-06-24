#!/usr/bin/env python3
"""Print a compact catchup report for repo-local .agent_state planning files."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _resolve_plan(root: Path, plan_id: str | None) -> Path | None:
    plans_root = root / ".agent_state" / "plans"
    if plan_id:
        candidate = plans_root / plan_id
        return candidate if candidate.is_dir() else None

    active = _read_text(root / ".agent_state" / "active_plan").strip()
    if active:
        candidate = plans_root / active
        if candidate.is_dir():
            return candidate

    if not plans_root.is_dir():
        return None

    candidates = [
        path
        for path in plans_root.iterdir()
        if path.is_dir() and (path / "task_plan.md").is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _head(text: str, lines: int) -> str:
    return "\n".join(text.splitlines()[:lines])


def _tail(text: str, lines: int) -> str:
    parts = text.splitlines()
    return "\n".join(parts[-lines:])


def _git_diff_stat(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "diff", "--stat"],
            cwd=root,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        return f"git diff --stat unavailable: {exc}"

    output = result.stdout.strip()
    if output:
        return output
    if result.stderr.strip():
        return result.stderr.strip()
    return "no unstaged tracked diff"


def _git_status_short(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=root,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        return f"git status --short unavailable: {exc}"

    output = result.stdout.strip()
    if output:
        return output
    if result.stderr.strip():
        return result.stderr.strip()
    return "clean"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--plan-id")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    plan_dir = _resolve_plan(root, args.plan_id)

    print("[planning-with-files] catchup")
    print(f"repo: {root}")

    if plan_dir is None:
        print("active plan: <none>")
        print('hint: run scripts/init-plan.sh <task-id> "任務目標"')
        return 0

    print(f"active plan: {plan_dir.name}")
    print(f"plan dir: {plan_dir}")
    print("")
    print("=== task_plan.md head ===")
    print(_head(_read_text(plan_dir / "task_plan.md"), 40))
    print("")
    print("=== recent progress ===")
    print(_tail(_read_text(plan_dir / "progress.md"), 20))
    print("")
    print("=== git diff --stat ===")
    print(_git_diff_stat(root))
    print("")
    print("=== git status --short ===")
    print(_git_status_short(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
