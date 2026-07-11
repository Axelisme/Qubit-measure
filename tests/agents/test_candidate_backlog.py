from __future__ import annotations

import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_ROOTS = [
    REPO_ROOT / ".agents" / "skills",
    REPO_ROOT / ".codex" / "skills",
    REPO_ROOT / ".claude" / "skills",
]
SCRIPT = SKILL_ROOTS[0] / "candidate-backlog" / "scripts" / "backlog.py"


def run_cli(
    root: Path, *args: str, ok: bool = True
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), *args],
        capture_output=True,
        check=False,
        text=True,
    )
    assert (result.returncode == 0) is ok, result.stderr
    return result


def add(root: Path, title: str = "Missing cancellation coverage") -> str:
    result = run_cli(
        root,
        "--json",
        "add",
        "--kind",
        "test-gap",
        "--area",
        "experiment-v2",
        "--source-task",
        "current-task",
        "--title",
        title,
        "--observation",
        "Cancellation has no focused test.",
        "--evidence",
        "Reviewer found the uncovered branch.",
        "--impact",
        "A regression could leak work.",
        "--desired-outcome",
        "Cancellation behavior has repeatable coverage.",
    )
    return json.loads(result.stdout)["id"]


def create_plan(root: Path, task_id: str) -> None:
    path = root / ".agent_state" / "plans" / task_id / "task_plan.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Task plan\n", encoding="utf-8")


def test_lifecycle_requires_planning_and_resolution_evidence(tmp_path: Path) -> None:
    item_id = add(tmp_path)
    failed = run_cli(
        tmp_path,
        "close",
        item_id,
        "--resolution",
        "implemented",
        ok=False,
    )
    assert "planned item" in failed.stderr

    create_plan(tmp_path, "future-task")
    run_cli(tmp_path, "plan", item_id, "--task-id", "future-task")
    failed = run_cli(
        tmp_path,
        "close",
        item_id,
        "--resolution",
        "implemented",
        "--task-id",
        "wrong-task",
        "--commit",
        "abc1234",
        "--validation",
        "pytest passed",
        ok=False,
    )
    assert "bound task-id" in failed.stderr

    run_cli(
        tmp_path,
        "close",
        item_id,
        "--resolution",
        "implemented",
        "--task-id",
        "future-task",
        "--commit",
        "abc1234",
        "--validation",
        "pytest passed",
    )
    payload = json.loads(
        run_cli(tmp_path, "list", "--status", "resolved", "--json").stdout
    )
    assert payload[0]["id"] == item_id
    assert payload[0]["validation"] == ["pytest passed"]


def test_plan_requires_existing_path_safe_formal_task(tmp_path: Path) -> None:
    item_id = add(tmp_path)
    missing = run_cli(tmp_path, "plan", item_id, "--task-id", "missing-task", ok=False)
    assert "formal task plan not found" in missing.stderr
    traversal = run_cli(tmp_path, "plan", item_id, "--task-id", "../escape", ok=False)
    assert "invalid task-id" in traversal.stderr


def test_duplicate_title_is_unicode_normalized_and_fast_fails(tmp_path: Path) -> None:
    item_id = add(tmp_path, "Ｆｏｏ   BAR")
    failed = run_cli(
        tmp_path,
        "add",
        "--kind",
        "defect",
        "--area",
        "gui",
        "--source-task",
        "task",
        "--title",
        "foo bar",
        "--observation",
        "x",
        "--evidence",
        "x",
        "--impact",
        "x",
        "--desired-outcome",
        "x",
        ok=False,
    )
    assert item_id in failed.stderr


def test_required_capture_field_rejects_whitespace(tmp_path: Path) -> None:
    failed = run_cli(
        tmp_path,
        "add",
        "--kind",
        "defect",
        "--area",
        "gui",
        "--source-task",
        "task",
        "--title",
        "Empty evidence",
        "--observation",
        "x",
        "--evidence",
        "   ",
        "--impact",
        "x",
        "--desired-outcome",
        "x",
        ok=False,
    )
    assert "evidence must not be empty" in failed.stderr


def test_duplicate_resolution_requires_existing_canonical_item(tmp_path: Path) -> None:
    item_id = add(tmp_path, "First finding")
    failed = run_cli(
        tmp_path,
        "close",
        item_id,
        "--resolution",
        "duplicate",
        "--duplicate-of",
        "BL-20260101T000000000000Z-missing",
        ok=False,
    )
    assert "not found" in failed.stderr


def test_utf8_and_filters_round_trip(tmp_path: Path) -> None:
    item_id = add(tmp_path, "取消流程缺少測試")
    payload = json.loads(
        run_cli(tmp_path, "list", "--area", "experiment-v2", "--json").stdout
    )
    assert payload[0]["id"] == item_id
    assert payload[0]["title"] == "取消流程缺少測試"


def test_manual_template_is_valid_for_full_lifecycle(tmp_path: Path) -> None:
    item_id = "BL-20000101T000000000000Z-replace-me"
    inbox = tmp_path / ".agent_state" / "backlog" / "inbox"
    inbox.mkdir(parents=True)
    template = SKILL_ROOTS[0] / "candidate-backlog" / "assets" / "item-template.md"
    (inbox / f"{item_id}.md").write_bytes(template.read_bytes())

    listed = json.loads(run_cli(tmp_path, "list", "--json").stdout)
    assert listed[0]["id"] == item_id
    create_plan(tmp_path, "template-task")
    run_cli(tmp_path, "plan", item_id, "--task-id", "template-task")
    run_cli(
        tmp_path,
        "close",
        item_id,
        "--resolution",
        "declined",
        "--note",
        "Example template lifecycle.",
    )
    assert (
        tmp_path / ".agent_state" / "backlog" / "closed" / f"{item_id}.md"
    ).is_file()


def test_concurrent_same_title_creates_exactly_one_item(tmp_path: Path) -> None:
    def invoke() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--root",
                str(tmp_path),
                "add",
                "--kind",
                "defect",
                "--area",
                "gui",
                "--source-task",
                "task",
                "--title",
                "Concurrent discovery",
                "--observation",
                "x",
                "--evidence",
                "x",
                "--impact",
                "x",
                "--desired-outcome",
                "x",
            ],
            capture_output=True,
            check=False,
            text=True,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: invoke(), range(2)))
    assert sorted(result.returncode for result in results) == [0, 40]
    assert (
        len(list((tmp_path / ".agent_state" / "backlog" / "inbox").glob("*.md"))) == 1
    )


def test_list_waits_for_transition_and_returns_one_snapshot(tmp_path: Path) -> None:
    spec = spec_from_file_location("candidate_backlog_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    item_id = add(tmp_path, "Transition snapshot")
    create_plan(tmp_path, "snapshot-task")

    destination_written = threading.Event()
    allow_unlink = threading.Event()
    original_atomic_write = module.atomic_write

    def pausing_atomic_write(path: Path, text: str) -> None:
        original_atomic_write(path, text)
        if path.parent.name == "planned":
            destination_written.set()
            assert allow_unlink.wait(timeout=5)

    setattr(module, "atomic_write", pausing_atomic_write)
    plan_args = SimpleNamespace(root=tmp_path, item_id=item_id, task_id="snapshot-task")
    list_args = SimpleNamespace(root=tmp_path, status=None, kind=None, area=None)
    with ThreadPoolExecutor(max_workers=2) as pool:
        transition = pool.submit(module.command_plan, plan_args)
        assert destination_written.wait(timeout=5)
        snapshot = pool.submit(module.command_list, list_args)
        assert not snapshot.done()
        allow_unlink.set()
        transition.result(timeout=5)
        items = snapshot.result(timeout=5)
    assert [(item["id"], item["status"]) for item in items] == [(item_id, "planned")]


@pytest.mark.parametrize("skill_name", ["orchestrate", "candidate-backlog"])
def test_skill_trees_stay_byte_identical(skill_name: str) -> None:
    trees = []
    for root in SKILL_ROOTS:
        skill = root / skill_name
        trees.append(
            {
                path.relative_to(skill): path.read_bytes()
                for path in skill.rglob("*")
                if path.is_file()
            }
        )
    assert trees[1:] == trees[:-1]


def test_orchestrate_main_skill_keeps_hard_gate_anchors() -> None:
    text = (SKILL_ROOTS[0] / "orchestrate" / "SKILL.md").read_text(encoding="utf-8")
    for anchor in (
        "`critical` diff 必須由不同 agent identity 獨立 review",
        "主 checkout preview/final 只走 merge queue",
        "commit / merge 只在使用者要求或授權後執行",
        "`light`",
        "`standard`",
        "`critical`",
        "lane-implementer",
        "integration-reviewer",
        "candidate-backlog",
        "設計先於實作",
    ):
        assert anchor in text
    for reference in (
        "worktree-protocol.md",
        "merge-protocol.md",
        "delegation-review.md",
        "validation.md",
        "state-contract.md",
        "merge-internals.md",
    ):
        assert (SKILL_ROOTS[0] / "orchestrate" / "references" / reference).is_file()


def test_repo_agents_routes_development_discoveries_only() -> None:
    text = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "DEVELOPMENT agent" in text
    assert "candidate-backlog" in text
    assert "不適用 MEASUREMENT 角色" in text
