"""Run the fast quality gates in one command.

With --base, run formatting on touched files, import contracts, script awaits,
MCP text budgets, registry key paths and the ratchet. Some checks scan the whole
repository or launch subprocesses, so runtime is not proportional only to the
changed file count.

Collection and behavioral tests stay separate: this command does not establish
that tests collect consistently or that the changed behavior passes. Runtime
depends on the worktree, installed environment and selected checks.

Order matters. Formatting runs first because it rewrites code, and measuring
before it would report a state that no longer exists by the time anyone reads the
result.

Without --base there is nothing to judge a candidate against, so the command
reports where the repository stands instead: the absolute count each check
produces over the whole tree. That is a different question from "did this change
make it worse", and it deserves a different answer rather than a silent fallback
to the merge-base with main -- which on a long-lived branch means nine hundred
changed files and several minutes of Pyright.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import _support

_TOOLS_DIR: Final = Path(__file__).resolve().parent

# The slow checks, named here so the help text can point at them rather than
# leaving a reader to assume this command covers everything.
SEPARATE_CHECKS: Final = (
    "python tools/check_pytest_collection.py",
    "pytest -n auto --dist=worksteal",
)


@dataclass(frozen=True)
class Check:
    """One absolute measurement of the whole tree."""

    name: str
    command: tuple[str, ...]
    # Where the count lives in the command's stdout. None means the exit code is
    # the whole answer.
    count_key: str | None = None


# Measured over the whole tree, not against a base. Pyright is opt-in because
# it adds a full-tree type-checking pass.
STATUS_CHECKS: Final = (
    Check("import contracts", ("lint-imports",)),
    Check(
        "script awaits",
        (sys.executable, str(_TOOLS_DIR / "check_script_awaits.py")),
        "violation_count",
    ),
    Check(
        "mcp text budget",
        (sys.executable, str(_TOOLS_DIR / "check_mcp_text_budget.py")),
        "overage_count",
    ),
    Check(
        "registry key paths",
        (sys.executable, str(_TOOLS_DIR / "check_registry_key_paths.py")),
        "violation_count",
    ),
    Check("ruff", ("ruff", "check", "--output-format", "json", "."), "__len__"),
    Check(
        "docstring",
        (
            "ruff",
            "check",
            "--isolated",
            "--select",
            "D100,D101,D102,D103",
            "--output-format",
            "json",
            "tools",
        ),
        "__len__",
    ),
    Check(
        "file size",
        (sys.executable, str(_TOOLS_DIR / "check_quality.py"), "all"),
        "file-size.violation_count",
    ),
    Check(
        "test paths",
        (sys.executable, str(_TOOLS_DIR / "check_quality.py"), "all"),
        "test-paths.violation_count",
    ),
    Check(
        "test structure",
        (sys.executable, str(_TOOLS_DIR / "check_quality.py"), "all"),
        "test-structure.violation_count",
    ),
    Check(
        "test capabilities",
        (sys.executable, str(_TOOLS_DIR / "check_quality.py"), "all"),
        "test-capabilities.violation_count",
    ),
    Check(
        "suppressions",
        (sys.executable, str(_TOOLS_DIR / "check_quality.py"), "all"),
        "suppressions.total",
    ),
)

PYRIGHT_CHECK: Final = Check(
    "pyright",
    ("pyright", *_support.PYRIGHT_THREADS, "--outputjson"),
    "summary.errorCount",
)


@dataclass(frozen=True)
class Step:
    """One named command in the gate's sequence."""

    name: str
    command: tuple[str, ...]


@dataclass(frozen=True)
class Outcome:
    """Captured gate-step result.

    name is the display label; command is the executed argv. returncode is the
    process exit status. stdout and stderr are captured text, including failures.
    """

    name: str
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        """Return whether the command exited with status zero."""
        return self.returncode == 0


def present(root: Path, files: tuple[str, ...]) -> tuple[str, ...]:
    """The changed files that still exist, which are all a rewriter can take.

    A deleted file is still a change the ratchet judges against the base, but
    handing its path to ruff fails the whole step.
    """
    return tuple(name for name in files if (root / name).is_file())


def steps(base: str, files: tuple[str, ...], *, fix: bool) -> tuple[Step, ...]:
    """Return the commands to run, in order.

    With no changed Python files there is nothing to format or lint, but the
    ratchet still runs: a candidate can widen a per-file-ignore without touching
    one line of Python.
    """
    found: list[Step] = []
    if files:
        import_flags = ("--fix",) if fix else ()
        format_flags = () if fix else ("--check",)
        found.append(
            Step(
                "ruff import sort",
                ("ruff", "check", "--select", "I", *import_flags, *files),
            )
        )
        found.append(Step("ruff format", ("ruff", "format", *format_flags, *files)))
    found.append(Step("import contracts", ("lint-imports",)))
    found.append(
        Step(
            "script awaits",
            (sys.executable, str(_TOOLS_DIR / "check_script_awaits.py")),
        )
    )
    found.append(
        Step(
            "mcp text budget",
            (sys.executable, str(_TOOLS_DIR / "check_mcp_text_budget.py")),
        )
    )
    found.append(
        Step(
            "registry key paths",
            (sys.executable, str(_TOOLS_DIR / "check_registry_key_paths.py")),
        )
    )
    found.append(
        Step(
            "ratchet",
            (sys.executable, str(_TOOLS_DIR / "check_ratchet.py"), "--base", base),
        )
    )
    return tuple(found)


def measure(check: Check, root: Path) -> str:
    """Return one check's absolute reading, as text for the status table."""
    completed = subprocess.run(
        check.command, cwd=root, check=False, capture_output=True, text=True
    )
    return _reading(check, completed)


def _reading(check: Check, completed: subprocess.CompletedProcess[str]) -> str:
    if check.count_key is None:
        return "green" if completed.returncode == 0 else "RED"
    try:
        payload = json.loads(completed.stdout or "null")
    except json.JSONDecodeError:
        return "unreadable"
    if check.count_key == "__len__":
        return str(len(payload)) if isinstance(payload, list) else "unreadable"
    for part in check.count_key.split("."):
        if not isinstance(payload, dict) or part not in payload:
            return "unreadable"
        payload = payload[part]
    return str(payload)


def status(root: Path, *, with_pyright: bool) -> dict[str, str]:
    """Return each check's absolute reading over the whole tree."""
    checks = (*STATUS_CHECKS, PYRIGHT_CHECK) if with_pyright else STATUS_CHECKS
    completed: dict[tuple[str, ...], subprocess.CompletedProcess[str]] = {}
    readings: dict[str, str] = {}
    for check in checks:
        if check.command not in completed:
            completed[check.command] = subprocess.run(
                check.command, cwd=root, check=False, capture_output=True, text=True
            )
        readings[check.name] = _reading(check, completed[check.command])
    return readings


def run_step(step: Step, root: Path) -> Outcome:
    """Run step.command in root and capture output without checking its status.

    Return the labeled result; subprocess launch/IO failures propagate.
    """
    completed = subprocess.run(
        step.command, cwd=root, check=False, capture_output=True, text=True
    )
    return Outcome(
        name=step.name,
        command=step.command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _ratchet_summary(outcome: Outcome) -> str:
    try:
        report = json.loads(outcome.stdout)
    except json.JSONDecodeError:
        return (
            outcome.stderr.strip().splitlines()[-1] if outcome.stderr else "no output"
        )
    lines: list[str] = []
    for name, detail in sorted(report.get("detectors", {}).items()):
        for item in detail["regressions"]:
            lines.append(
                f"    {name}: {item['path']} {item['rule']} "
                f"{item['before']} -> {item['after']}"
            )
    changed = len(report.get("changed_files", []))
    head = f"{changed} changed Python file(s), base {report.get('base', '?')[:9]}"
    return "\n".join([head, *lines])


def main(argv: list[str] | None = None) -> int:
    """Report absolute counts, or run this checkout's gates against --base.

    None reads sys.argv. Without --base, print counts and return 0, regardless
    of findings; --with-pyright includes a full-tree type pass. With --base,
    rewrite touched Python files unless --no-fix, then run the remaining gates.
    Return 0 if all steps pass, 1 for a failed step, or 2 if the base cannot be
    resolved. argparse handles help and invalid CLI syntax with SystemExit;
    subprocess launch failures propagate.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Run separately: " + "; ".join(SEPARATE_CHECKS),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base",
        default=None,
        help="commit to judge against; omit to report the tree's absolute counts",
    )
    parser.add_argument(
        "--no-fix",
        action="store_true",
        help="do not rewrite files; only judge them",
    )
    parser.add_argument(
        "--with-pyright",
        action="store_true",
        help="include a whole-tree Pyright pass in the status report",
    )
    arguments = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]

    if arguments.base is None:
        readings = status(root, with_pyright=arguments.with_pyright)
        width = max(len(name) for name in readings)
        for name, reading in readings.items():
            print(f"{name.ljust(width)}  {reading}")
        if not arguments.with_pyright:
            print("\npyright omitted; --with-pyright adds a full-tree type pass")
        print("These are absolute counts over the whole tree, not a verdict on any")
        print("change. Pass --base <ref> to judge a candidate against its base.")
        return 0

    ratchet = _support.load_tool("check_ratchet")
    try:
        base = ratchet.resolve_base(root, arguments.base)
        files = present(root, ratchet.changed_python_files(root, base))
    except ratchet.RatchetError as error:
        print(f"gate failed: {error}", file=sys.stderr)
        return 2

    failed = False
    for step in steps(base, files, fix=not arguments.no_fix):
        outcome = run_step(step, root)
        mark = "ok  " if outcome.ok else "FAIL"
        print(f"{mark} {outcome.name}")
        if outcome.name == "ratchet" and outcome.stdout:
            try:
                changes = json.loads(outcome.stdout).get("configuration_changes", [])
            except json.JSONDecodeError:
                changes = []
            for change in changes:
                print(
                    f"REVIEW {change['path']}: "
                    f"{json.dumps(change['before'])} -> {json.dumps(change['after'])}"
                )
        if outcome.ok:
            continue
        failed = True
        detail = (
            _ratchet_summary(outcome)
            if outcome.name == "ratchet"
            else (outcome.stdout or outcome.stderr).strip()
        )
        print(detail)
    if failed:
        print("\nRun separately: " + "; ".join(SEPARATE_CHECKS), file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
