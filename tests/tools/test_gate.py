from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import Mock

import gate
import pytest


def test_formatting_runs_before_anything_measures_the_code() -> None:
    """Measuring first would report a state the formatter is about to replace."""
    names = [step.name for step in gate.steps("abc", ("lib/a.py",), fix=True)]

    assert names == ["ruff import sort", "ruff format", "import contracts", "ratchet"]


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


class _StubRatchet:
    """Stands in for the loaded check_ratchet tool: a fixed base, one changed file."""

    class RatchetError(Exception):
        pass

    @staticmethod
    def resolve_base(_root: Path, base: str) -> str:
        return base

    @staticmethod
    def changed_python_files(_root: Path, _base: str) -> tuple[str, ...]:
        return ()


def _run_gate_with_ratchet_outcome(
    monkeypatch: pytest.MonkeyPatch, ratchet: gate.Outcome
) -> tuple[int, list[str]]:
    monkeypatch.setattr(gate._support, "load_tool", lambda _name: _StubRatchet)

    def fake_run_step(step: gate.Step, _root: Path) -> gate.Outcome:
        if step.name == "ratchet":
            return ratchet
        return gate.Outcome(step.name, step.command, 0, "", "")

    monkeypatch.setattr(gate, "run_step", fake_run_step)
    lines: list[str] = []
    monkeypatch.setattr(
        "builtins.print", lambda *a, **_k: lines.append(" ".join(map(str, a)))
    )
    code = gate.main(["--base", "abcdef1234", "--no-fix"])
    return code, lines


def test_a_failing_ratchet_names_the_rule_that_rose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    code, output = _run_gate_with_ratchet_outcome(monkeypatch, outcome)

    assert code == 1
    assert "FAIL ratchet" in output
    joined = "\n".join(output)
    assert "lib/a.py C901 1 -> 2" in joined
    assert "1 changed Python file(s)" in joined


def test_an_unreadable_ratchet_report_falls_back_to_its_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcome = gate.Outcome(
        name="ratchet",
        command=("x",),
        returncode=2,
        stdout="",
        stderr="ratchet failed: git merge-base failed\n",
    )

    code, output = _run_gate_with_ratchet_outcome(monkeypatch, outcome)

    assert code == 1
    assert output.index("FAIL ratchet") + 1 == output.index(
        "ratchet failed: git merge-base failed"
    )


@pytest.mark.parametrize(
    ("stdout", "returncode", "count_key", "expected"),
    [
        ('{"violation_count": 19}', 0, "violation_count", "19"),
        ('{"summary": {"errorCount": 3620}}', 1, "summary.errorCount", "3620"),
        ("[1, 2, 3]", 1, "__len__", "3"),
        ("", 0, None, "green"),
        ("", 1, None, "RED"),
        ("not json", 0, "total", "unreadable"),
    ],
    ids=["named-key", "nested-key", "list-length", "success", "failure", "unreadable"],
)
def test_measure_reports_external_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stdout: str,
    returncode: int,
    count_key: str | None,
    expected: str,
) -> None:
    check = gate.Check("probe", ("external-check",), count_key)
    receipt = subprocess.CompletedProcess(check.command, returncode, stdout, "")
    monkeypatch.setattr(gate.subprocess, "run", Mock(return_value=receipt))

    assert gate.measure(check, tmp_path) == expected
