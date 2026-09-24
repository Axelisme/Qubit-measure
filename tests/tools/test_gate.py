from __future__ import annotations

import sys
from pathlib import Path

import gate


def test_formatting_runs_before_anything_measures_the_code() -> None:
    """Measuring first would report a state the formatter is about to replace."""
    names = [step.name for step in gate.steps("abc", ("lib/a.py",), fix=True)]

    assert names == ["ruff import sort", "ruff format", "import contracts", "ratchet"]


def test_the_rewriting_steps_can_be_turned_off() -> None:
    names = [step.name for step in gate.steps("abc", ("lib/a.py",), fix=False)]

    assert names == ["import contracts", "ratchet"]


def test_the_ratchet_runs_even_when_no_python_file_changed() -> None:
    """A candidate can widen a per-file-ignore without touching any Python."""
    names = [step.name for step in gate.steps("abc", (), fix=True)]

    assert names == ["import contracts", "ratchet"]


def test_the_rewriting_steps_receive_only_the_changed_files() -> None:
    found = gate.steps("abc", ("lib/a.py", "tests/b.py"), fix=True)

    assert found[0].command[-2:] == ("lib/a.py", "tests/b.py")
    assert found[1].command[-2:] == ("lib/a.py", "tests/b.py")


def test_deleted_files_are_not_handed_to_the_rewriters(tmp_path: Path) -> None:
    """A retirement candidate deletes files; ruff fails on a path that is gone."""
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "kept.py").write_text("x = 1\n")

    assert gate.present(tmp_path, ("lib/gone.py", "lib/kept.py")) == ("lib/kept.py",)


def test_the_ratchet_is_given_the_resolved_base() -> None:
    ratchet = next(
        step for step in gate.steps("deadbeef", (), fix=True) if step.name == "ratchet"
    )

    assert ratchet.command[-2:] == ("--base", "deadbeef")


def test_the_slow_checks_are_named_rather_than_folded_in() -> None:
    """Folding a two-minute suite into a two-second command teaches people to skip it."""
    commands = [step.command for step in gate.steps("abc", ("lib/a.py",), fix=True)]
    joined = " ".join(part for command in commands for part in command)

    assert "pytest" not in joined
    assert any("pytest" in separate for separate in gate.SEPARATE_CHECKS)
    assert any("check_pytest_collection" in s for s in gate.SEPARATE_CHECKS)


def test_a_regression_summary_names_the_rule_that_rose() -> None:
    outcome = gate.Outcome(
        name="ratchet",
        command=("x",),
        returncode=1,
        stdout=(
            '{"base": "abcdef1234", "changed_files": ["lib/a.py"], "detectors": '
            '{"ruff": {"regression_count": 1, "regressions": [{"after": 2, '
            '"before": 1, "path": "lib/a.py", "rule": "C901"}]}}, "status": "FAIL"}'
        ),
        stderr="",
    )

    summary = gate._ratchet_summary(outcome)

    assert "lib/a.py C901 1 -> 2" in summary
    assert "1 changed Python file(s)" in summary


def test_an_unreadable_ratchet_report_falls_back_to_its_error() -> None:
    outcome = gate.Outcome(
        name="ratchet",
        command=("x",),
        returncode=2,
        stdout="",
        stderr="ratchet failed: git merge-base failed\n",
    )

    assert gate._ratchet_summary(outcome) == "ratchet failed: git merge-base failed"


def test_a_json_receipt_reading_comes_from_its_named_key(tmp_path: Path):
    check = gate.Check(
        "probe",
        (sys.executable, "-c", "print('{\"violation_count\": 19}')"),
        "violation_count",
    )

    assert gate.measure(check, tmp_path) == "19"


def test_a_nested_key_is_followed(tmp_path: Path):
    check = gate.Check(
        "probe",
        (sys.executable, "-c", 'print(\'{"summary": {"errorCount": 3620}}\')'),
        "summary.errorCount",
    )

    assert gate.measure(check, tmp_path) == "3620"


def test_a_json_list_is_read_by_length(tmp_path: Path):
    """Ruff emits an array of diagnostics; its own summary lines are not data."""
    check = gate.Check("probe", (sys.executable, "-c", "print('[1,2,3]')"), "__len__")

    assert gate.measure(check, tmp_path) == "3"


def test_a_check_without_a_count_key_reports_its_exit_code(tmp_path: Path):
    green = gate.Check("probe", (sys.executable, "-c", "pass"))
    red = gate.Check("probe", (sys.executable, "-c", "raise SystemExit(1)"))

    assert gate.measure(green, tmp_path) == "green"
    assert gate.measure(red, tmp_path) == "RED"


def test_unreadable_output_is_reported_rather_than_guessed(tmp_path: Path):
    check = gate.Check("probe", (sys.executable, "-c", "print('not json')"), "total")

    assert gate.measure(check, tmp_path) == "unreadable"


def test_pyright_is_not_in_the_default_status_set() -> None:
    """41 seconds against 6 would make the status view something people stop running."""
    assert all(check.name != "pyright" for check in gate.STATUS_CHECKS)
    assert gate.PYRIGHT_CHECK.name == "pyright"
