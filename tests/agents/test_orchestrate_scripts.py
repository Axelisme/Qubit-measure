from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_SCRIPT = (
    REPO_ROOT / ".agents" / "skills" / "orchestrate" / "scripts" / "workflow.py"
)
CODEX_WORKFLOW_SCRIPT = (
    REPO_ROOT / ".codex" / "skills" / "orchestrate" / "scripts" / "workflow.py"
)
CLAUDE_WORKFLOW_SCRIPT = (
    REPO_ROOT / ".claude" / "skills" / "orchestrate" / "scripts" / "workflow.py"
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


def run_git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"git {' '.join(args)} failed with {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}",
        )
    return result


def git_stdout(root: Path, *args: str) -> str:
    return run_git(root, *args).stdout.strip()


def git_returncode(root: Path, *args: str) -> int:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    ).returncode


def init_git_repo(root: Path) -> str:
    run_git(root, "init")
    run_git(root, "checkout", "-b", "main")
    run_git(root, "config", "user.email", "agent@example.invalid")
    run_git(root, "config", "user.name", "Agent Test")
    (root / ".gitignore").write_text(".agent_state/\nscratch/\n", encoding="utf-8")
    (root / "tracked.txt").write_text("base\n", encoding="utf-8")
    run_git(root, "add", ".gitignore", "tracked.txt")
    run_git(root, "commit", "-m", "base")
    return git_stdout(root, "rev-parse", "HEAD")


def create_integration_branch(root: Path, task_id: str = "demo-task") -> str:
    run_git(root, "checkout", "-b", f"agent/{task_id}")
    (root / "tracked.txt").write_text(f"{task_id}\n", encoding="utf-8")
    run_git(root, "add", "tracked.txt")
    run_git(root, "commit", "-m", task_id)
    target = git_stdout(root, "rev-parse", "HEAD")
    run_git(root, "checkout", "main")
    return target


def create_task_and_lane(
    root: Path,
    task_id: str = "demo-task",
    *,
    base_commit: str = "abc123",
) -> None:
    run_script(WORKFLOW_SCRIPT, "--root", str(root), "state", "init")
    run_script(
        WORKFLOW_SCRIPT,
        "--root",
        str(root),
        "task",
        "create",
        task_id,
        "--base-branch",
        "main",
        "--base-commit",
        base_commit,
        "--integration-branch",
        f"agent/{task_id}",
    )
    run_script(
        WORKFLOW_SCRIPT,
        "--root",
        str(root),
        "lane",
        "create",
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


def queue_entry(
    root: Path,
    task_id: str,
    *,
    action: str,
    base_commit: str,
    status: str,
    target_commit: str,
) -> dict[str, Any]:
    return {
        "action": action,
        "base_branch": "main",
        "base_head": base_commit,
        "branch": f"agent/{task_id}",
        "enqueued_at": "2026-07-03T00:00:00Z",
        "main_worktree": root.as_posix(),
        "note": "",
        "requested_by": "agent-a",
        "started_at": None,
        "status": status,
        "target_commit": target_commit,
        "task_id": task_id,
        "token": None,
    }


def write_queue(root: Path, entries: list[dict[str, Any]]) -> None:
    queue_file = root / ".agent_state" / "worktrees" / "merge_queue.json"
    queue_file.write_text(
        json.dumps({"version": 1, "queue": entries}, sort_keys=True),
        encoding="utf-8",
    )


def read_task(root: Path, task_id: str = "demo-task") -> dict[str, Any]:
    shown = run_script(
        WORKFLOW_SCRIPT,
        "--root",
        str(root),
        "task",
        "show",
        task_id,
        "--json",
    )
    return json.loads(shown.stdout)["task"]


def test_workflow_manages_task_lane_and_report_path(tmp_path: Path) -> None:
    create_task_and_lane(tmp_path)

    rerun = run_script(
        WORKFLOW_SCRIPT,
        "--root",
        str(tmp_path),
        "task",
        "create",
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
        WORKFLOW_SCRIPT,
        "--root",
        str(tmp_path),
        "report",
        "path",
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

    task = read_task(tmp_path)
    assert task["worktrees"]["main"]["worktree_path"] == (
        ".agent_state/worktrees/trees/demo-task"
    )
    assert task["worktrees"]["main"]["write_scope"] == ["lib/..."]


def test_preview_start_and_abort_manage_queue_and_task_status(tmp_path: Path) -> None:
    base_commit = init_git_repo(tmp_path)
    target_commit = create_integration_branch(tmp_path)
    create_task_and_lane(tmp_path, base_commit=base_commit)

    started = run_script(
        WORKFLOW_SCRIPT,
        "--root",
        str(tmp_path),
        "preview",
        "start",
        "demo-task",
        "--requested-by",
        "agent-a",
        "--json",
    )
    entry = json.loads(started.stdout)["entry"]
    assert entry["action"] == "preview"
    assert entry["status"] == "merging"
    assert entry["target_commit"] == target_commit
    assert (tmp_path / ".git" / "MERGE_HEAD").read_text(encoding="utf-8").strip() == (
        target_commit
    )
    assert read_task(tmp_path)["status"] == "merge_preview"

    run_script(
        WORKFLOW_SCRIPT,
        "--root",
        str(tmp_path),
        "preview",
        "abort",
        "demo-task",
    )
    assert not (tmp_path / ".git" / "MERGE_HEAD").exists()
    assert read_task(tmp_path)["status"] == "reviewing"

    queue = run_script(
        WORKFLOW_SCRIPT,
        "--root",
        str(tmp_path),
        "queue",
        "list",
        "--json",
    )
    assert json.loads(queue.stdout)["queue"]["queue"] == []


def test_preview_allows_nonoverlapping_untracked_file(tmp_path: Path) -> None:
    base_commit = init_git_repo(tmp_path)
    create_integration_branch(tmp_path)
    create_task_and_lane(tmp_path, base_commit=base_commit)
    (tmp_path / "report.md").write_text("local report\n", encoding="utf-8")

    run_script(
        WORKFLOW_SCRIPT,
        "--root",
        str(tmp_path),
        "preview",
        "start",
        "demo-task",
        "--requested-by",
        "agent-a",
    )
    run_script(
        WORKFLOW_SCRIPT,
        "--root",
        str(tmp_path),
        "preview",
        "abort",
        "demo-task",
    )

    assert (tmp_path / "report.md").read_text(encoding="utf-8") == "local report\n"
    assert read_task(tmp_path)["status"] == "reviewing"


def test_preview_rejects_untracked_file_collision(tmp_path: Path) -> None:
    base_commit = init_git_repo(tmp_path)
    run_git(tmp_path, "checkout", "-b", "agent/demo-task")
    (tmp_path / "report.md").write_text("branch report\n", encoding="utf-8")
    run_git(tmp_path, "add", "report.md")
    run_git(tmp_path, "commit", "-m", "add report")
    run_git(tmp_path, "checkout", "main")
    (tmp_path / "report.md").write_text("local report\n", encoding="utf-8")
    create_task_and_lane(tmp_path, base_commit=base_commit)

    result = run_script(
        WORKFLOW_SCRIPT,
        "--root",
        str(tmp_path),
        "preview",
        "start",
        "demo-task",
        "--requested-by",
        "agent-a",
        check=False,
    )

    assert result.returncode == 40
    assert "untracked files overlap the merge target" in result.stderr
    assert "report.md" in result.stderr
    assert not (tmp_path / ".git" / "MERGE_HEAD").exists()
    assert read_task(tmp_path)["status"] == "active"


def test_preview_rejects_ignored_untracked_file_collision(tmp_path: Path) -> None:
    base_commit = init_git_repo(tmp_path)
    run_git(tmp_path, "checkout", "-b", "agent/demo-task")
    (tmp_path / "scratch").mkdir()
    (tmp_path / "scratch" / "note.md").write_text("branch note\n", encoding="utf-8")
    run_git(tmp_path, "add", "-f", "scratch/note.md")
    run_git(tmp_path, "commit", "-m", "add ignored-path note")
    run_git(tmp_path, "checkout", "main")
    (tmp_path / "scratch").mkdir()
    (tmp_path / "scratch" / "note.md").write_text("local note\n", encoding="utf-8")
    create_task_and_lane(tmp_path, base_commit=base_commit)

    result = run_script(
        WORKFLOW_SCRIPT,
        "--root",
        str(tmp_path),
        "preview",
        "start",
        "demo-task",
        "--requested-by",
        "agent-a",
        check=False,
    )

    assert result.returncode == 40
    assert "untracked files overlap the merge target" in result.stderr
    assert "scratch/note.md" in result.stderr
    assert (tmp_path / "scratch" / "note.md").read_text(encoding="utf-8") == (
        "local note\n"
    )
    assert not (tmp_path / ".git" / "MERGE_HEAD").exists()


def test_preview_rejects_untracked_prefix_collision(tmp_path: Path) -> None:
    base_commit = init_git_repo(tmp_path)
    run_git(tmp_path, "checkout", "-b", "agent/demo-task")
    (tmp_path / "scratch").mkdir()
    (tmp_path / "scratch" / "note.md").write_text("branch note\n", encoding="utf-8")
    run_git(tmp_path, "add", "-f", "scratch/note.md")
    run_git(tmp_path, "commit", "-m", "add nested note")
    run_git(tmp_path, "checkout", "main")
    (tmp_path / "scratch").write_text("local file blocks directory\n", encoding="utf-8")
    create_task_and_lane(tmp_path, base_commit=base_commit)

    result = run_script(
        WORKFLOW_SCRIPT,
        "--root",
        str(tmp_path),
        "preview",
        "start",
        "demo-task",
        "--requested-by",
        "agent-a",
        check=False,
    )

    assert result.returncode == 40
    assert "untracked files overlap the merge target" in result.stderr
    assert "scratch" in result.stderr
    assert (tmp_path / "scratch").read_text(encoding="utf-8") == (
        "local file blocks directory\n"
    )
    assert not (tmp_path / ".git" / "MERGE_HEAD").exists()


def test_preview_start_requires_base_branch(tmp_path: Path) -> None:
    base_commit = init_git_repo(tmp_path)
    create_integration_branch(tmp_path)
    create_task_and_lane(tmp_path, base_commit=base_commit)
    run_git(tmp_path, "checkout", "-b", "scratch")

    result = run_script(
        WORKFLOW_SCRIPT,
        "--root",
        str(tmp_path),
        "preview",
        "start",
        "demo-task",
        "--requested-by",
        "agent-a",
        check=False,
    )

    assert result.returncode == 40
    assert "expected base branch main" in result.stderr
    assert not (tmp_path / ".git" / "MERGE_HEAD").exists()
    assert read_task(tmp_path)["status"] == "active"


def test_preview_start_rejects_blocked_queue_entry(tmp_path: Path) -> None:
    base_commit = init_git_repo(tmp_path)
    target_commit = create_integration_branch(tmp_path)
    create_task_and_lane(tmp_path, base_commit=base_commit)
    write_queue(
        tmp_path,
        [
            {
                **queue_entry(
                    tmp_path,
                    "demo-task",
                    action="preview",
                    base_commit=base_commit,
                    status="blocked",
                    target_commit=target_commit,
                ),
                "note": "manual block",
            }
        ],
    )

    result = run_script(
        WORKFLOW_SCRIPT,
        "--root",
        str(tmp_path),
        "preview",
        "start",
        "demo-task",
        "--requested-by",
        "agent-a",
        check=False,
    )

    assert result.returncode == 40
    assert "queue entry is blocked" in result.stderr
    queue = json.loads(
        (tmp_path / ".agent_state" / "worktrees" / "merge_queue.json").read_text(
            encoding="utf-8",
        )
    )["queue"]
    assert queue[0]["status"] == "blocked"
    assert queue[0]["note"] == "manual block"


def test_final_fast_forward_aborts_preview_and_closes_task(tmp_path: Path) -> None:
    base_commit = init_git_repo(tmp_path)
    target_commit = create_integration_branch(tmp_path)
    create_task_and_lane(tmp_path, base_commit=base_commit)
    run_script(
        WORKFLOW_SCRIPT,
        "--root",
        str(tmp_path),
        "preview",
        "start",
        "demo-task",
        "--requested-by",
        "agent-a",
    )

    run_script(
        WORKFLOW_SCRIPT,
        "--root",
        str(tmp_path),
        "final",
        "fast-forward",
        "demo-task",
        "--requested-by",
        "agent-a",
    )

    assert git_stdout(tmp_path, "rev-parse", "HEAD") == target_commit
    missing = run_script(
        WORKFLOW_SCRIPT,
        "--root",
        str(tmp_path),
        "task",
        "show",
        "demo-task",
        check=False,
    )
    assert missing.returncode == 40
    queue = run_script(
        WORKFLOW_SCRIPT,
        "--root",
        str(tmp_path),
        "queue",
        "list",
        "--json",
    )
    assert json.loads(queue.stdout)["queue"]["queue"] == []


def test_final_fast_forward_allows_nonoverlapping_untracked_file(
    tmp_path: Path,
) -> None:
    base_commit = init_git_repo(tmp_path)
    target_commit = create_integration_branch(tmp_path)
    create_task_and_lane(tmp_path, base_commit=base_commit)
    (tmp_path / "report.md").write_text("local report\n", encoding="utf-8")

    run_script(
        WORKFLOW_SCRIPT,
        "--root",
        str(tmp_path),
        "final",
        "fast-forward",
        "demo-task",
        "--requested-by",
        "agent-a",
    )

    assert git_stdout(tmp_path, "rev-parse", "HEAD") == target_commit
    assert (tmp_path / "report.md").read_text(encoding="utf-8") == "local report\n"


def test_final_fast_forward_waits_behind_queue_head(tmp_path: Path) -> None:
    base_commit = init_git_repo(tmp_path)
    first_target = create_integration_branch(tmp_path, "first-task")
    second_target = create_integration_branch(tmp_path, "second-task")
    create_task_and_lane(tmp_path, "first-task", base_commit=base_commit)
    create_task_and_lane(tmp_path, "second-task", base_commit=base_commit)
    queue_file = tmp_path / ".agent_state" / "worktrees" / "merge_queue.json"
    write_queue(
        tmp_path,
        [
            queue_entry(
                tmp_path,
                "first-task",
                action="final",
                base_commit=base_commit,
                status="queued",
                target_commit=first_target,
            )
        ],
    )

    result = run_script(
        WORKFLOW_SCRIPT,
        "--root",
        str(tmp_path),
        "final",
        "fast-forward",
        "second-task",
        "--requested-by",
        "agent-b",
        check=False,
    )

    assert result.returncode == 20
    assert "queued behind first-task" in result.stderr
    queue = json.loads(queue_file.read_text(encoding="utf-8"))["queue"]
    assert [entry["task_id"] for entry in queue] == ["first-task", "second-task"]
    assert queue[1]["target_commit"] == second_target


def test_queue_validation_rejects_duplicate_task_ids(tmp_path: Path) -> None:
    base_commit = init_git_repo(tmp_path)
    target_commit = create_integration_branch(tmp_path)
    create_task_and_lane(tmp_path, base_commit=base_commit)
    write_queue(
        tmp_path,
        [
            queue_entry(
                tmp_path,
                "demo-task",
                action="preview",
                base_commit=base_commit,
                status="queued",
                target_commit=target_commit,
            ),
            queue_entry(
                tmp_path,
                "demo-task",
                action="final",
                base_commit=base_commit,
                status="queued",
                target_commit=target_commit,
            ),
        ],
    )

    result = run_script(
        WORKFLOW_SCRIPT,
        "--root",
        str(tmp_path),
        "queue",
        "list",
        check=False,
    )

    assert result.returncode == 10
    assert "duplicate task_id demo-task" in result.stderr


def test_worktree_create_lane_records_generated_worktree(tmp_path: Path) -> None:
    base_commit = init_git_repo(tmp_path)

    created = run_script(
        WORKFLOW_SCRIPT,
        "--root",
        str(tmp_path),
        "worktree",
        "create-lane",
        "demo-task",
        "main",
        "--base-branch",
        "main",
        "--write-scope",
        "lib/...",
        "--json",
    )

    result = json.loads(created.stdout)
    worktree_path = Path(result["worktree_path"])
    assert worktree_path.is_dir()
    assert (worktree_path / ".git").exists()
    assert result["branch"] == "agent/demo-task"
    task = read_task(tmp_path)
    assert task["base_commit"] == base_commit
    assert task["worktrees"]["main"]["worktree_path"] == (
        ".agent_state/worktrees/trees/demo-task"
    )


def test_worktree_create_lane_preflights_state_conflict(tmp_path: Path) -> None:
    init_git_repo(tmp_path)
    run_script(WORKFLOW_SCRIPT, "--root", str(tmp_path), "state", "init")
    run_script(
        WORKFLOW_SCRIPT,
        "--root",
        str(tmp_path),
        "task",
        "create",
        "conflict-task",
        "--base-branch",
        "main",
        "--base-commit",
        "different-base",
        "--integration-branch",
        "agent/conflict-task",
    )

    result = run_script(
        WORKFLOW_SCRIPT,
        "--root",
        str(tmp_path),
        "worktree",
        "create-lane",
        "conflict-task",
        "main",
        "--base-branch",
        "main",
        "--write-scope",
        "lib/...",
        check=False,
    )

    assert result.returncode == 40
    assert "different identity" in result.stderr
    assert (
        git_returncode(
            tmp_path,
            "show-ref",
            "--verify",
            "--quiet",
            "refs/heads/agent/conflict-task",
        )
        == 1
    )
    assert not (
        tmp_path / ".agent_state" / "worktrees" / "trees" / "conflict-task"
    ).exists()


def test_codex_workflow_script_is_self_contained(tmp_path: Path) -> None:
    run_script(CODEX_WORKFLOW_SCRIPT, "--root", str(tmp_path), "state", "init")
    state_file = tmp_path / ".agent_state" / "worktrees" / "state.json"
    queue_file = tmp_path / ".agent_state" / "worktrees" / "merge_queue.json"
    assert json.loads(state_file.read_text(encoding="utf-8")) == {
        "version": 2,
        "tasks": {},
    }
    assert json.loads(queue_file.read_text(encoding="utf-8")) == {
        "version": 1,
        "queue": [],
    }


def test_claude_workflow_script_is_self_contained(tmp_path: Path) -> None:
    run_script(CLAUDE_WORKFLOW_SCRIPT, "--root", str(tmp_path), "state", "init")
    state_file = tmp_path / ".agent_state" / "worktrees" / "state.json"
    queue_file = tmp_path / ".agent_state" / "worktrees" / "merge_queue.json"
    assert json.loads(state_file.read_text(encoding="utf-8")) == {
        "version": 2,
        "tasks": {},
    }
    assert json.loads(queue_file.read_text(encoding="utf-8")) == {
        "version": 1,
        "queue": [],
    }
