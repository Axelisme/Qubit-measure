from __future__ import annotations

import io
import json
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import check_ratchet as ratchet
import pytest


@dataclass
class CommandReceipts:
    """External Git/Ruff outputs; the ratchet still parses and compares them."""

    changed: str = "lib/widen.py\n"
    untracked: str = ""
    before_ruff: int = 0
    after_ruff: int = 0


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "lib").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        '[tool.ruff.lint]\nextend-select = ["PLR0913"]\n\n'
        "[tool.ruff.lint.pylint]\nmax-args = 6\n",
        encoding="utf-8",
    )
    (root / "lib" / "widen.py").write_text(
        "def widen(value: int) -> int:\n    return value + 1\n", encoding="utf-8"
    )
    return root


@pytest.fixture
def command_receipts(
    repository: Path, monkeypatch: pytest.MonkeyPatch
) -> CommandReceipts:
    receipts = CommandReceipts()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name in ("pyproject.toml", "lib/widen.py"):
            archive.add(repository / name, arcname=name)
    base_archive = buffer.getvalue()

    def respond(
        command: tuple[str, ...],
        *,
        cwd: Path,
        stdout: BinaryIO | None = None,
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        if command[0] == "ruff":
            count = receipts.after_ruff if cwd == repository else receipts.before_ruff
            diagnostics = [
                {"filename": str(cwd / "lib/widen.py"), "code": "PLR0913"}
                for _ in range(count)
            ]
            return subprocess.CompletedProcess(
                command, int(count > 0), json.dumps(diagnostics), ""
            )
        if command == ("git", "archive", "--format=tar", "HEAD"):
            assert stdout is not None
            stdout.write(base_archive)
            return subprocess.CompletedProcess(command, 0, "", "")
        outputs: dict[tuple[str, ...], str] = {
            ("git", "rev-parse", "HEAD"): "HEAD\n",
            ("git", "diff", "--name-only", "HEAD"): receipts.changed,
            ("git", "ls-files", "--others", "--exclude-standard"): receipts.untracked,
            (
                "git",
                "diff",
                "--no-ext-diff",
                "--name-status",
                "-z",
                "--find-renames=20%",
                "HEAD",
                "--",
            ): "",
        }
        return subprocess.CompletedProcess(command, 0, outputs[command], "")

    monkeypatch.setattr(ratchet.subprocess, "run", respond)
    return receipts


def test_a_count_that_rises_is_a_regression() -> None:
    found = ratchet.regressions({("a.py", "C901"): 1}, {("a.py", "C901"): 2})

    assert [(item.path, item.before, item.after) for item in found] == [("a.py", 1, 2)]


def test_an_unchanged_or_falling_count_is_not_a_regression() -> None:
    before = {("a.py", "C901"): 2, ("b.py", "C901"): 3}
    after = {("a.py", "C901"): 2, ("b.py", "C901"): 1}

    assert ratchet.regressions(before, after) == ()


def test_a_rule_absent_from_the_base_counts_from_zero() -> None:
    found = ratchet.regressions({}, {("new.py", "ERA001"): 1})

    assert found[0].before == 0
    assert found[0].increase == 1


def test_regressions_are_ordered_by_size_of_increase() -> None:
    after = {("a.py", "R"): 2, ("b.py", "R"): 9, ("c.py", "R"): 5}

    found = ratchet.regressions({}, after)

    assert [item.path for item in found] == ["b.py", "c.py", "a.py"]


def test_changed_files_cover_the_working_tree_not_only_commits(
    repository: Path, command_receipts: CommandReceipts
) -> None:
    command_receipts.untracked = "lib/untracked.py\n"

    found = ratchet.changed_python_files(repository, "HEAD")

    assert found == ("lib/untracked.py", "lib/widen.py")


def test_files_outside_the_checked_roots_are_ignored(
    repository: Path, command_receipts: CommandReceipts
) -> None:
    command_receipts.changed = "docs/note.py\nlib/notes.md\n"

    assert ratchet.changed_python_files(repository, "HEAD") == ()


def test_the_base_tree_is_measured_with_the_candidate_configuration(
    repository: Path, command_receipts: CommandReceipts
) -> None:
    """Enabling a rule must not make its existing findings look new."""
    (repository / "pyproject.toml").write_text("[tool.ruff]\n", encoding="utf-8")
    destination = repository.parent / "scratch"
    destination.mkdir()

    ratchet.extract_base_tree(repository, "HEAD", destination)

    extracted = (destination / "tree" / "pyproject.toml").read_text(encoding="utf-8")
    assert extracted == "[tool.ruff]\n"


def test_adding_a_violation_fails_and_names_the_rule(
    repository: Path, command_receipts: CommandReceipts
) -> None:
    command_receipts.after_ruff = 1

    report = ratchet.run(repository, "HEAD", ["ruff"])

    assert report["status"] == "FAIL"
    assert report["detectors"]["ruff"]["regressions"] == [
        {"path": "lib/widen.py", "rule": "PLR0913", "before": 0, "after": 1}
    ]


def test_a_change_that_adds_no_violation_passes(
    repository: Path, command_receipts: CommandReceipts
) -> None:
    report = ratchet.run(repository, "HEAD", ["ruff"])

    assert report["status"] == "PASS"


def test_removing_a_violation_passes(
    repository: Path, command_receipts: CommandReceipts
) -> None:
    command_receipts.before_ruff = 1

    report = ratchet.run(repository, "HEAD", ["ruff"])

    assert report["status"] == "PASS"


def test_the_base_tree_keeps_its_own_configuration_alongside(
    repository: Path, command_receipts: CommandReceipts
) -> None:
    """Substituting the candidate's config must not hide a new per-file-ignore."""
    original = (repository / "pyproject.toml").read_text(encoding="utf-8")
    (repository / "pyproject.toml").write_text("[tool.ruff]\n", encoding="utf-8")
    destination = repository.parent / "scratch"
    destination.mkdir()

    ratchet.extract_base_tree(repository, "HEAD", destination)

    tree = destination / "tree"
    assert (tree / "pyproject.toml").read_text(encoding="utf-8") == "[tool.ruff]\n"
    assert (tree / "pyproject.base.toml").read_text(encoding="utf-8") == original


def test_widening_a_per_file_ignore_is_a_regression(
    repository: Path, command_receipts: CommandReceipts
) -> None:
    command_receipts.changed = "pyproject.toml\nlib/widen.py\n"
    (repository / "pyproject.toml").write_text(
        '[tool.ruff.lint]\nextend-select = ["PLR0913"]\n\n'
        "[tool.ruff.lint.pylint]\nmax-args = 6\n\n"
        "[tool.ruff.lint.per-file-ignores]\n"
        '"lib/**" = ["PLR0913", "C901"]\n',
        encoding="utf-8",
    )

    report = ratchet.run(repository, "HEAD", ["suppressions"])

    assert report["status"] == "FAIL"
    regressions = report["detectors"]["suppressions"]["regressions"]
    assert regressions[0]["rule"] == "suppression:ruff-per-file-ignore"
    assert regressions[0]["before"] == 0
    assert regressions[0]["after"] == 2


def test_a_candidate_that_changes_only_configuration_is_still_judged(
    repository: Path, command_receipts: CommandReceipts
) -> None:
    """No Python file changes, so the detectors must not be skipped."""
    command_receipts.changed = "pyproject.toml\n"
    (repository / "pyproject.toml").write_text(
        '[tool.ruff.lint]\nignore = ["E402", "E501"]\n', encoding="utf-8"
    )

    report = ratchet.run(repository, "HEAD", ["suppressions"])

    assert report["changed_files"] == []
    assert report["status"] == "FAIL"
