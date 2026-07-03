from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_SCRIPT = REPO_ROOT / ".agents" / "skills" / "orchestrate" / "scripts" / "state.py"
QUEUE_SCRIPT = (
    REPO_ROOT / ".agents" / "skills" / "orchestrate" / "scripts" / "merge_queue.py"
)
CODEX_STATE_SCRIPT = (
    REPO_ROOT / ".codex" / "skills" / "orchestrate" / "scripts" / "state.py"
)
CLAUDE_STATE_SCRIPT = (
    REPO_ROOT / ".claude" / "skills" / "orchestrate" / "scripts" / "state.py"
)


def run_script(
    script: Path,
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, str(script), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise AssertionError(
            f"{script.name} failed with {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}",
        )
    return result


def create_task_and_lane(root: Path, task_id: str = "demo-task") -> None:
    run_script(STATE_SCRIPT, "--root", str(root), "init")
    run_script(
        STATE_SCRIPT,
        "--root",
        str(root),
        "task-create",
        task_id,
        "--base-branch",
        "main",
        "--base-commit",
        "abc123",
        "--integration-branch",
        f"agent/{task_id}",
    )
    run_script(
        STATE_SCRIPT,
        "--root",
        str(root),
        "lane-create",
        task_id,
        "main",
        "--role",
        "lane",
        "--branch",
        f"agent/{task_id}",
        "--worktree-path",
        f".agent_state/worktrees/trees/{task_id}",
        "--reports-dir",
        f".agent_state/worktrees/reports/{task_id}/main",
        "--write-scope",
        "lib/...",
    )


def test_state_script_manages_task_lane_and_report_path(tmp_path: Path) -> None:
    create_task_and_lane(tmp_path)

    rerun = run_script(
        STATE_SCRIPT,
        "--root",
        str(tmp_path),
        "task-create",
        "demo-task",
        "--base-branch",
        "main",
        "--base-commit",
        "abc123",
        "--integration-branch",
        "agent/demo-task",
    )
    assert rerun.returncode == 0

    report = run_script(
        STATE_SCRIPT,
        "--root",
        str(tmp_path),
        "report-path",
        "demo-task",
        "main",
        "planner",
        "--mkdir",
        "--json",
    )
    report_path = Path(json.loads(report.stdout)["report_path"])
    assert report_path == (
        tmp_path
        / ".agent_state"
        / "worktrees"
        / "reports"
        / "demo-task"
        / "main"
        / "planner.md"
    )
    assert report_path.parent.is_dir()

    shown = run_script(
        STATE_SCRIPT,
        "--root",
        str(tmp_path),
        "task-show",
        "demo-task",
        "--json",
    )
    task = json.loads(shown.stdout)["task"]
    assert task["worktrees"]["main"]["worktree_path"] == (
        ".agent_state/worktrees/trees/demo-task"
    )
    assert task["worktrees"]["main"]["write_scope"] == ["lib/..."]


def test_merge_queue_claim_release_and_task_status_sync(tmp_path: Path) -> None:
    create_task_and_lane(tmp_path)
    run_script(QUEUE_SCRIPT, "--root", str(tmp_path), "init")
    run_script(
        QUEUE_SCRIPT,
        "--root",
        str(tmp_path),
        "enqueue",
        "demo-task",
        "--branch",
        "agent/demo-task",
        "--base-branch",
        "main",
        "--action",
        "preview",
        "--requested-by",
        "agent-a",
    )
    run_script(QUEUE_SCRIPT, "--root", str(tmp_path), "claim", "demo-task")
    run_script(QUEUE_SCRIPT, "--root", str(tmp_path), "assert-held", "demo-task")

    claimed = run_script(
        STATE_SCRIPT,
        "--root",
        str(tmp_path),
        "task-show",
        "demo-task",
        "--json",
    )
    assert json.loads(claimed.stdout)["task"]["status"] == "merge_preview"

    run_script(
        QUEUE_SCRIPT,
        "--root",
        str(tmp_path),
        "release",
        "demo-task",
        "--result",
        "preview-aborted",
    )
    released = run_script(
        STATE_SCRIPT,
        "--root",
        str(tmp_path),
        "task-show",
        "demo-task",
        "--json",
    )
    assert json.loads(released.stdout)["task"]["status"] == "reviewing"


def test_merge_queue_enforces_head_claim(tmp_path: Path) -> None:
    create_task_and_lane(tmp_path, "first-task")
    create_task_and_lane(tmp_path, "second-task")
    run_script(QUEUE_SCRIPT, "--root", str(tmp_path), "init")
    for task_id in ("first-task", "second-task"):
        run_script(
            QUEUE_SCRIPT,
            "--root",
            str(tmp_path),
            "enqueue",
            task_id,
            "--branch",
            f"agent/{task_id}",
            "--base-branch",
            "main",
            "--action",
            "preview",
            "--requested-by",
            "agent-a",
        )

    result = run_script(
        QUEUE_SCRIPT,
        "--root",
        str(tmp_path),
        "claim",
        "second-task",
        check=False,
    )
    assert result.returncode == 20
    assert "not queue head" in result.stderr


def test_codex_wrapper_invokes_primary_state_script(tmp_path: Path) -> None:
    run_script(CODEX_STATE_SCRIPT, "--root", str(tmp_path), "init")
    state_file = tmp_path / ".agent_state" / "worktrees" / "state.json"
    assert json.loads(state_file.read_text(encoding="utf-8")) == {
        "version": 2,
        "tasks": {},
    }


def test_claude_state_script_is_self_contained(tmp_path: Path) -> None:
    run_script(CLAUDE_STATE_SCRIPT, "--root", str(tmp_path), "init")
    state_file = tmp_path / ".agent_state" / "worktrees" / "state.json"
    assert json.loads(state_file.read_text(encoding="utf-8")) == {
        "version": 2,
        "tasks": {},
    }
