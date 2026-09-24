from __future__ import annotations

import subprocess
from pathlib import Path

import check_ratchet as ratchet
import pytest

# The integration cases below drive a real git repository through real child
# processes, which is the capability this marker exists to declare.
pytestmark = pytest.mark.requires_subprocess

_CLEAN_SOURCE = "def widen(value: int) -> int:\n    return value + 1\n"
# Seven parameters trips PLR0913, whose threshold this repository sets to six.
_REGRESSED_SOURCE = (
    "def widen(a: int, b: int, c: int, d: int, e: int, f: int, g: int) -> int:\n"
    "    return a + b + c + d + e + f + g\n"
)


def _git(root: Path, *args: str) -> None:
    subprocess.run(("git", *args), cwd=root, check=True, capture_output=True)


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    """A git repository whose lib/widen.py is committed and clean."""
    root = tmp_path / "repo"
    (root / "lib").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        '[tool.ruff.lint]\nextend-select = ["PLR0913"]\n\n'
        "[tool.ruff.lint.pylint]\nmax-args = 6\n",
        encoding="utf-8",
    )
    (root / "lib" / "widen.py").write_text(_CLEAN_SOURCE, encoding="utf-8")
    _git(root, "init", "--quiet")
    _git(root, "config", "user.email", "ratchet@example.invalid")
    _git(root, "config", "user.name", "ratchet")
    _git(root, "add", "-A")
    _git(root, "commit", "--quiet", "-m", "base")
    return root


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


def test_changed_files_cover_the_working_tree_not_only_commits(repository: Path):
    (repository / "lib" / "widen.py").write_text(_REGRESSED_SOURCE, encoding="utf-8")
    (repository / "lib" / "untracked.py").write_text(_CLEAN_SOURCE, encoding="utf-8")

    found = ratchet.changed_python_files(repository, "HEAD")

    assert found == ("lib/untracked.py", "lib/widen.py")


def test_files_outside_the_checked_roots_are_ignored(repository: Path):
    (repository / "docs").mkdir()
    (repository / "docs" / "note.py").write_text(_CLEAN_SOURCE, encoding="utf-8")
    (repository / "lib" / "notes.md").write_text("text\n", encoding="utf-8")

    assert ratchet.changed_python_files(repository, "HEAD") == ()


def test_the_base_tree_is_measured_with_the_candidate_configuration(repository: Path):
    """Enabling a rule must not make its existing findings look new."""
    (repository / "pyproject.toml").write_text("[tool.ruff]\n", encoding="utf-8")
    destination = repository.parent / "scratch"
    destination.mkdir()

    ratchet.extract_base_tree(repository, "HEAD", destination)

    extracted = (destination / "tree" / "pyproject.toml").read_text(encoding="utf-8")
    assert extracted == "[tool.ruff]\n"


def test_adding_a_violation_fails_and_names_the_rule(repository: Path):
    (repository / "lib" / "widen.py").write_text(_REGRESSED_SOURCE, encoding="utf-8")

    report = ratchet.run(repository, "HEAD", ["ruff"])

    assert report["status"] == "FAIL"
    assert report["detectors"]["ruff"]["regressions"][0]["rule"] == "PLR0913"


def test_a_change_that_adds_no_violation_passes(repository: Path):
    (repository / "lib" / "widen.py").write_text(
        _CLEAN_SOURCE + "\n\ndef narrow(value: int) -> int:\n    return value - 1\n",
        encoding="utf-8",
    )

    report = ratchet.run(repository, "HEAD", ["ruff"])

    assert report["status"] == "PASS"


def test_removing_a_violation_passes(repository: Path):
    (repository / "lib" / "widen.py").write_text(_REGRESSED_SOURCE, encoding="utf-8")
    _git(repository, "add", "-A")
    _git(repository, "commit", "--quiet", "-m", "regressed")
    (repository / "lib" / "widen.py").write_text(_CLEAN_SOURCE, encoding="utf-8")

    report = ratchet.run(repository, "HEAD", ["ruff"])

    assert report["status"] == "PASS"


def test_the_base_tree_keeps_its_own_configuration_alongside(repository: Path):
    """Substituting the candidate's config must not hide a new per-file-ignore."""
    original = (repository / "pyproject.toml").read_text(encoding="utf-8")
    (repository / "pyproject.toml").write_text("[tool.ruff]\n", encoding="utf-8")
    destination = repository.parent / "scratch"
    destination.mkdir()

    ratchet.extract_base_tree(repository, "HEAD", destination)

    tree = destination / "tree"
    assert (tree / "pyproject.toml").read_text(encoding="utf-8") == "[tool.ruff]\n"
    assert (tree / "pyproject.base.toml").read_text(encoding="utf-8") == original


def test_widening_a_per_file_ignore_is_a_regression(repository: Path):
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


def test_a_candidate_that_changes_only_configuration_is_still_judged(repository: Path):
    """No Python file changes, so the detectors must not be skipped."""
    (repository / "pyproject.toml").write_text(
        '[tool.ruff.lint]\nignore = ["E402", "E501"]\n', encoding="utf-8"
    )

    report = ratchet.run(repository, "HEAD", ["suppressions"])

    assert report["changed_files"] == []
    assert report["status"] == "FAIL"
