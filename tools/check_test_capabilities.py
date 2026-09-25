"""Check that a test module declares the capabilities it uses.

Slow and flaky tests come from constructs, not from duration: real sockets, real
subprocesses, and sleeps that trade wall-clock for a missing synchronisation
point. A duration threshold measures the symptom and can be escaped by marking a
test slow. This measures the cause, and its markers cannot be traded away --
they name a capability, so mismarking makes the test wrong rather than exempt.

Detection is static, deliberately. The runtime alternative -- patching
socket.socket, subprocess.Popen and time.sleep from an autouse fixture -- would
reach into a suite that runs under pytest-xdist and drives Qt, where those same
primitives carry the test harness itself. A guard that can break every test in
the suite is a poor trade for catching indirect uses; the direct ones are where
the problem is.

The unit is the module. Statically there is no honest way to tie a sleep inside
a helper to the test that reaches it, so a module that uses a capability must
declare it -- via module-level `pytestmark` or a marker on one of its tests.

Call sites are resolved through the module's imports rather than matched on the
text of the attribute chain. `from time import sleep` and `import time as t` are
ordinary Python, not evasion, and a check that only recognised `time.sleep` would
report neither -- leaving the honest spelling as the only one that gets caught.
"""

from __future__ import annotations

import ast
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final

import _support

# capability -> (marker that declares it, fully qualified callables that use it)
CAPABILITIES: Final = {
    "wall-clock": ("uses_wall_clock", ("time.sleep", "asyncio.sleep")),
    "loopback": (
        "requires_loopback",
        (
            "socket.socket",
            "socket.create_connection",
            "socket.create_server",
            "socket.socketpair",
        ),
    ),
    "subprocess": (
        "requires_subprocess",
        (
            "subprocess.run",
            "subprocess.Popen",
            "subprocess.call",
            "subprocess.check_call",
            "subprocess.check_output",
            "os.system",
            "os.popen",
            "multiprocessing.Process",
        ),
    ),
}

_TESTS_DIR: Final = "tests"


@dataclass(frozen=True)
class UndeclaredCapability:
    """A test module that uses a capability without declaring its marker."""

    module: PurePosixPath
    capability: str
    marker: str
    uses: int


def capability_uses(tree: ast.AST) -> dict[str, int]:
    """Return {capability: number of call sites} for one parsed module."""
    wanted = {
        dotted: capability
        for capability, (_, dotted_names) in CAPABILITIES.items()
        for dotted in dotted_names
    }
    bindings = _support.import_bindings(tree)
    found: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        qualified = _support.resolve_call(node.func, bindings)
        capability = wanted.get(qualified) if qualified else None
        if capability is not None:
            found[capability] = found.get(capability, 0) + 1
    return found


def declared_markers(tree: ast.AST) -> set[str]:
    """Return every pytest marker named anywhere in one parsed module."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        dotted = _support.dotted_name(node)
        if dotted is not None and dotted.startswith("pytest.mark."):
            found.add(dotted.removeprefix("pytest.mark."))
    return found


def undeclared_capabilities(
    root: Path, *, paths: Iterable[str] | None = None
) -> tuple[UndeclaredCapability, ...]:
    """Return every test module using a capability it does not declare.

    `paths` narrows the walk to named files. The unit here is the module, and a
    module only gains a capability by being edited, so the ratchet passes the
    candidate's changed files rather than re-reading every test in the tree.
    """
    tests = root / _TESTS_DIR
    if not tests.is_dir():
        return ()
    if paths is None:
        candidates = sorted(tests.rglob("test_*.py"))
    else:
        candidates = [
            root / name
            for name in sorted(paths)
            if name.startswith(f"{_TESTS_DIR}/")
            and name.rpartition("/")[2].startswith("test_")
        ]
    found: list[UndeclaredCapability] = []
    for path in candidates:
        if not path.is_file():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        markers = declared_markers(tree)
        for capability, uses in sorted(capability_uses(tree).items()):
            marker = CAPABILITIES[capability][0]
            if marker in markers:
                continue
            found.append(
                UndeclaredCapability(
                    module=PurePosixPath(path.relative_to(root).as_posix()),
                    capability=capability,
                    marker=marker,
                    uses=uses,
                )
            )
    return tuple(found)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    found = undeclared_capabilities(root)
    receipt = {
        "capabilities": {name: marker for name, (marker, _) in CAPABILITIES.items()},
        "status": "FAIL" if found else "PASS",
        "violation_count": len(found),
        "violations": [
            {
                "capability": item.capability,
                "marker": item.marker,
                "module": str(item.module),
                "uses": item.uses,
            }
            for item in found
        ],
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))
    for item in found:
        print(
            f"{item.module}: uses {item.capability} "
            f"({item.uses} call sites) without @pytest.mark.{item.marker}",
            file=sys.stderr,
        )
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
