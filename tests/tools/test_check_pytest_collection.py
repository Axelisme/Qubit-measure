from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import check_pytest_collection as oracle
import pytest


def test_parse_collection_requires_success_summary_and_exact_node_count() -> None:
    command = ("python", "-m", "pytest")
    output = (
        "tests/a/test_same.py::test_first\n"
        "tests/b/test_same.py::test_second\n"
        "2 tests collected in 0.01s\n"
    )

    result = oracle.parse_collection(
        command=command, returncode=0, stdout=output, stderr=""
    )

    assert result.node_ids == (
        "tests/a/test_same.py::test_first",
        "tests/b/test_same.py::test_second",
    )
    assert len(result.digest) == 64


@pytest.mark.parametrize("entrypoint", ["module", "console"])
def test_collection_commands_use_the_same_venv_entrypoint(
    entrypoint: str,
) -> None:
    root = Path("/repo")

    serial = oracle.build_command(root, parallel=False, entrypoint=entrypoint)
    parallel = oracle.build_command(root, parallel=True, entrypoint=entrypoint)

    assert serial[-1] == parallel[-1] == "/repo/tests"
    assert "-n" not in serial
    assert parallel[-3:-1] == ("-n", "auto")
    if entrypoint == "module":
        assert serial[:3] == (sys.executable, "-m", "pytest")
    else:
        console_name = "pytest.exe" if sys.platform == "win32" else "pytest"
        assert serial[0] == str(Path(sys.executable).with_name(console_name))


@pytest.mark.parametrize(
    ("platform", "console_name"), [("linux", "pytest"), ("win32", "pytest.exe")]
)
def test_console_entrypoint_uses_platform_venv_name(
    monkeypatch: pytest.MonkeyPatch, platform: str, console_name: str
) -> None:
    monkeypatch.setattr(oracle.sys, "platform", platform)

    command = oracle.build_command(Path("/repo"), parallel=False, entrypoint="console")

    assert command[0] == str(Path(sys.executable).with_name(console_name))


def test_collect_removes_inherited_pythonpath_for_both_entrypoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path("/repo")
    calls: list[tuple[tuple[str, ...], dict[str, str]]] = []

    def fake_run(
        command: tuple[str, ...],
        *,
        cwd: Path,
        env: dict[str, str],
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert cwd == root
        assert check is False
        assert capture_output is True
        assert text is True
        assert "PYTHONPATH" not in env
        calls.append((command, env))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="tests/a.py::test_a\n1 test collected in 0.01s\n",
            stderr="",
        )

    monkeypatch.setenv("PYTHONPATH", "/foreign/path")
    monkeypatch.setattr(oracle.subprocess, "run", fake_run)

    for entrypoint in ("module", "console"):
        oracle.collect(root, parallel=False, entrypoint=entrypoint)

    console_name = "pytest.exe" if sys.platform == "win32" else "pytest"
    assert [call[0][0] for call in calls] == [
        sys.executable,
        str(Path(sys.executable).with_name(console_name)),
    ]


def test_collection_matrix_has_one_result_per_entrypoint_and_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, bool]] = []
    expected = oracle.CollectionResult(
        ("pytest",), ("tests/a.py::test_a",), "1 test collected in 0.01s"
    )

    def fake_collect(_root: Path, *, parallel: bool, entrypoint: str) -> object:
        calls.append((entrypoint, parallel))
        return expected

    monkeypatch.setattr(oracle, "collect", fake_collect)
    monkeypatch.setattr(
        oracle, "configured_addopts", lambda _root: ("--import-mode=importlib",)
    )

    assert oracle.main() == 0
    assert set(calls) == {
        ("module", False),
        ("module", True),
        ("console", False),
        ("console", True),
    }


@pytest.mark.parametrize(
    ("returncode", "stdout", "match"),
    [
        (2, "collection error\n", "collection exited 2"),
        (0, "tests/a/test_x.py::test_x\n", "exactly one pytest summary"),
        (0, "1 test collected in 0.01s\n", "no test node ids"),
        (
            0,
            "tests/a/test_x.py::test_x\n2 tests collected in 0.01s\n",
            "summary count does not match",
        ),
        (
            0,
            "tests/a/test_x.py::test_x\n"
            "tests/a/test_x.py::test_x\n"
            "2 tests collected in 0.01s\n",
            "duplicate test node ids",
        ),
    ],
)
def test_parse_collection_rejects_inconclusive_evidence(
    returncode: int, stdout: str, match: str
) -> None:
    with pytest.raises(oracle.CollectionOracleError, match=match):
        oracle.parse_collection(
            command=("pytest",), returncode=returncode, stdout=stdout, stderr=""
        )


def test_compare_collections_rejects_set_drift_in_any_matrix_member() -> None:
    baseline = oracle.CollectionResult(
        ("module", "serial"), ("tests/a.py::test_a",), ""
    )
    same = oracle.CollectionResult(("module", "parallel"), ("tests/a.py::test_a",), "")
    drift = oracle.CollectionResult(("console", "serial"), ("tests/b.py::test_b",), "")

    with pytest.raises(oracle.CollectionOracleError, match="collection differ"):
        oracle.compare_collections(baseline, same, drift)


def test_collection_runs_with_colour_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A terminal exporting FORCE_COLOR would otherwise break summary parsing."""
    seen: list[dict[str, str]] = []

    def fake_run(command: tuple[str, ...], **kwargs: object) -> object:
        env = kwargs["env"]
        assert isinstance(env, dict)
        seen.append(env)
        coloured = "FORCE_COLOR" in env or "NO_COLOR" not in env
        summary = "1 test collected in 0.01s"
        if coloured:
            summary = f"\x1b[32m{summary}\x1b[0m"
        return subprocess.CompletedProcess(
            command, 0, stdout=f"tests/a.py::test_a\n{summary}\n", stderr=""
        )

    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(oracle.subprocess, "run", fake_run)

    result = oracle.collect(tmp_path, parallel=False)

    assert len(seen) == 1
    assert "FORCE_COLOR" not in seen[0]
    assert seen[0]["NO_COLOR"] == "1"
    assert "\x1b[" not in result.summary
    assert result.summary == "1 test collected in 0.01s"
