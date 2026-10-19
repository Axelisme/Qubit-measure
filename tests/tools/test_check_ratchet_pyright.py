"""Shared full-tree Pyright baseline cache and receipt contracts."""

from __future__ import annotations

import io
import json
import platform
import subprocess
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import check_ratchet as ratchet
import pytest


@dataclass
class PyrightCommands:
    """Synthetic Git/Pyright boundary; all paths stay inside the test sandbox."""

    root: Path
    common: Path
    base: str = "a" * 40
    version: str = "pyright 1.1.411\n"
    before: int = 1
    after: int = 1
    failure: bool = False
    changed: str = "lib/source.py\n"
    checks: list[tuple[Path, tuple[str, ...]]] = field(default_factory=list)
    archives: int = 0
    ruff_checks: list[Path] = field(default_factory=list)

    def run(
        self,
        command: tuple[str, ...],
        *,
        cwd: Path,
        stdout: BinaryIO | None = None,
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        """Return synthetic tool reports and an in-memory base Git archive."""
        output = ""
        code = 0
        if command[:2] == ("pyright", "--version"):
            output = self.version
        elif command[0] == "pyright":
            self.checks.append((cwd, command))
            count = self.after if cwd == self.root else self.before
            output = json.dumps(
                {
                    "generalDiagnostics": [
                        {
                            "file": str(cwd / "lib/source.py"),
                            "severity": "error",
                            "rule": "reportArgumentType",
                        }
                        for _ in range(count)
                    ]
                    + [{"file": "lib/source.py", "severity": "warning"}]
                }
            )
            code = 2 if self.failure else int(count > 0)
        elif command[:2] == ("git", "archive"):
            self.archives += 1
            assert stdout is not None
            with io.BytesIO() as buffer:
                with tarfile.open(fileobj=buffer, mode="w") as archive:
                    for name in ("pyproject.toml", "uv.lock", "lib/source.py"):
                        archive.add(self.common.parent / name, arcname=name)
                stdout.write(buffer.getvalue())
        elif command[0] == "git" and "--git-common-dir" in command:
            output = str(self.common) + "\n"
        elif command[:2] == ("git", "rev-parse"):
            output = self.base + "\n"
        elif command[:3] == ("git", "diff", "--name-only"):
            output = self.changed
        elif command[:2] == ("git", "ls-files") or "--name-status" in command:
            output = ""
        elif command[0] == "ruff":
            self.ruff_checks.append(cwd)
            output = "[]"
        else:
            raise AssertionError(f"Unexpected command: {command}")
        return subprocess.CompletedProcess(command, code, output, "synthetic failure")

    def compare(self, *, base: str = "HEAD") -> dict[str, object]:
        """Compare whole-tree reports through the public ratchet interface."""
        return ratchet.run(self.root, base, ("pyright",), full_pyright=True)

    @property
    def base_checks(self) -> int:
        """Number of actual baseline Pyright subprocess requests."""
        return sum(cwd != self.root for cwd, _command in self.checks)


@pytest.fixture
def commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PyrightCommands:
    root = tmp_path / "main"
    (root / "lib").mkdir(parents=True)
    (root / "lib/source.py").write_text("value = 1\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[tool.pyright]\n", encoding="utf-8")
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    common = root / ".git"
    common.mkdir()
    boundary = PyrightCommands(root, common)
    monkeypatch.setattr(subprocess, "run", boundary.run)
    return boundary


def test_full_comparison_reuses_counts_but_always_checks_candidate(
    commands: PyrightCommands,
) -> None:
    cold = commands.compare()
    hot = commands.compare()
    assert commands.base_checks == 1
    assert len(commands.checks) == 3
    assert commands.archives == 1
    assert cold.get("pyright_base") == "computed"
    assert hot.get("pyright_base") == "cache"
    assert cold["status"] == hot["status"] == "PASS"
    assert cold["base"] == commands.base
    assert cold["changed_files"] == ["lib/source.py"]
    assert cold["pyright_scope"] == "full"
    assert cold["configuration_changes"] == []
    assert (
        cold["detectors"]
        == hot["detectors"]
        == {"pyright": {"regression_count": 0, "regressions": []}}
    )
    for _ in range(2):
        changed = ratchet.run(commands.root, "HEAD", ("pyright",))
        assert changed["pyright_scope"] == "changed"
        assert changed["detectors"] == cold["detectors"]
    assert commands.base_checks == 3
    assert commands.archives == 3


@pytest.mark.parametrize("component", ["base", "config", "lock", "pyright", "python"])
def test_each_cache_key_component_invalidates_the_baseline(
    commands: PyrightCommands, monkeypatch: pytest.MonkeyPatch, component: str
) -> None:
    commands.compare()
    commands.compare()
    if component == "base":
        commands.base = "b" * 40
    elif component == "config":
        (commands.root / "pyproject.toml").write_text(
            "[tool.pyright]\ntypeCheckingMode = 'strict'\n", encoding="utf-8"
        )
    elif component == "lock":
        (commands.root / "uv.lock").write_text("version = 2\n", encoding="utf-8")
    elif component == "pyright":
        commands.version = "pyright 1.1.412\n"
    else:
        monkeypatch.setattr(platform, "python_version", lambda: "99.0.0")
    changed = commands.compare()
    assert commands.base_checks == 2
    assert changed.get("pyright_base") == "computed"


@pytest.mark.parametrize(
    "corrupt",
    [
        b"not json",
        b"\xff",
        b"null",
        b"{}",
        b"[]",
        b'[["lib/source.py", "rule", -1]]',
        b'[["lib/source.py", "rule", true]]',
    ],
)
def test_corrupt_cache_is_recomputed_and_replaced(
    commands: PyrightCommands, corrupt: bytes
) -> None:
    commands.compare()
    cached = list(
        (commands.common.parent / ".agent_state/ratchet/cache").glob("*.json")
    )
    assert len(cached) == 1
    cached[0].write_bytes(corrupt)
    repaired = commands.compare()
    hot = commands.compare()
    assert repaired.get("pyright_base") == "computed"
    assert hot.get("pyright_base") == "cache"
    assert commands.base_checks == 2
    assert json.loads(cached[0].read_text(encoding="utf-8")) is not None


@pytest.mark.parametrize("first", ["main", "lane"])
def test_worktrees_share_main_checkout_cache(
    commands: PyrightCommands, tmp_path: Path, first: str
) -> None:
    main = commands.root
    lane = tmp_path / "nested/worktrees/lane"
    (lane / "lib").mkdir(parents=True)
    for name in ("lib/source.py", "pyproject.toml", "uv.lock"):
        (lane / name).write_bytes((main / name).read_bytes())
    commands.root = main if first == "main" else lane
    commands.compare()
    cached = list((main / ".agent_state/ratchet/cache").glob("*.json"))
    assert len(cached) == 1
    commands.root = lane if first == "main" else main
    commands.checks.clear()
    report = commands.compare(base="same-commit-alias")
    assert report.get("pyright_base") == "cache"
    assert commands.checks and len(commands.checks) == 1
    assert commands.checks[0][0] == commands.root
    assert commands.archives == 1


@pytest.mark.parametrize("configuration_changed", [False, True])
def test_cache_hit_keeps_other_detectors_and_configuration_review(
    commands: PyrightCommands, configuration_changed: bool
) -> None:
    commands.compare()
    if configuration_changed:
        commands.changed += "pyproject.toml\n"
    report = ratchet.run(commands.root, "HEAD", ("pyright", "ruff"), full_pyright=True)
    assert commands.base_checks == 1
    assert commands.archives == 2
    assert len(commands.ruff_checks) == 2
    assert report.get("pyright_base") == "cache"
    assert report["configuration_changes"] == []
    assert report["detectors"] == {
        "pyright": {"regression_count": 0, "regressions": []},
        "ruff": {"regression_count": 0, "regressions": []},
    }


@pytest.mark.parametrize("after, expected", [(0, 0), (2, 1)])
def test_cached_baseline_preserves_main_receipt_and_exit_status(
    commands: PyrightCommands,
    capsys: pytest.CaptureFixture[str],
    after: int,
    expected: int,
) -> None:
    # main uses its own checkout, while the Git boundary keeps cache writes isolated.
    commands.root = Path(ratchet.__file__).resolve().parents[1]
    args = ["--base", "HEAD", "--detector", "pyright", "--full-pyright"]
    assert ratchet.main(args) == 0
    capsys.readouterr()
    commands.after = after
    assert ratchet.main(args) == expected
    report = json.loads(capsys.readouterr().out)
    assert commands.base_checks == 1
    assert report.get("pyright_base") == "cache"
    if expected:
        assert report["detectors"]["pyright"]["regressions"] == [
            {
                "path": "lib/source.py",
                "rule": "reportArgumentType",
                "before": 1,
                "after": 2,
            }
        ]


def test_failed_candidate_on_cache_hit_keeps_tool_error_exit(
    commands: PyrightCommands, capsys: pytest.CaptureFixture[str]
) -> None:
    commands.root = Path(ratchet.__file__).resolve().parents[1]
    args = ["--base", "HEAD", "--detector", "pyright", "--full-pyright"]
    assert ratchet.main(args) == 0
    capsys.readouterr()
    commands.failure = True
    assert ratchet.main(args) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "pyright failed" in captured.err
    assert commands.base_checks == 1
    assert len(commands.checks) == 3
