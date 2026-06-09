"""Invariants for the shared GUI layer (`zcu_tools.gui.remote.*`, `version_table`).

The shared transport mechanism and the optimistic-concurrency version table are
extracted from both GUI apps (measure-gui + fluxdep). They MUST stay app-agnostic
and import-clean: importing them pulls in neither Qt nor matplotlib nor any
``zcu_tools.gui.app.*`` package — otherwise the "shared" layer would secretly
depend on an app and the extraction would be a lie.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_SHARED_MODULES = [
    "zcu_tools.gui.remote.errors",
    "zcu_tools.gui.remote.framing",
    "zcu_tools.gui.remote.param_spec",
    "zcu_tools.gui.remote.wire",
    "zcu_tools.gui.version_table",
    # The session core (shared by measure + autofluxdep) must stay import-clean:
    # session value types, the session event vocabulary, and the async-operation
    # handle facet pull in neither Qt, matplotlib, nor any app package.
    "zcu_tools.gui.session.types",
    "zcu_tools.gui.session.events",
    "zcu_tools.gui.session.operation_handles",
    # The plotting package + its backend-select module must stay import-clean so
    # an entry script can configure the matplotlib backend before any pyplot
    # import. The heavy plotting submodules (backend/host/container) DO pull in
    # qtpy/matplotlib by design and are intentionally NOT listed here.
    "zcu_tools.gui.plotting",
    "zcu_tools.gui.plotting.setup",
]


@pytest.mark.parametrize("module", _SHARED_MODULES)
def test_shared_module_is_qt_and_matplotlib_clean(module: str) -> None:
    """Importing a shared module must not pull in Qt, matplotlib, or any app.

    Run in a subprocess because the pytest process has already imported Qt.
    """
    repo_root = Path(__file__).resolve().parents[2]
    script = textwrap.dedent(
        f"""
        import sys
        import {module}  # noqa: F401

        forbidden = ("matplotlib", "qtpy", "PyQt6", "PyQt5", "PySide6", "PySide2")
        leaked = sorted(
            m for m in sys.modules
            if m.split(".")[0] in forbidden
            or m.startswith("zcu_tools.gui.app.")
        )
        assert not leaked, f"importing {module} leaked: {{leaked}}"
        print("ok")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_version_table_is_single_shared_source() -> None:
    """Both apps' ``state.VersionTable`` resolve to the one shared class."""
    from zcu_tools.gui.app.fluxdep.state import VersionTable as FluxVT
    from zcu_tools.gui.app.main.state import VersionTable as MainVT
    from zcu_tools.gui.version_table import VersionTable as SharedVT

    assert MainVT is SharedVT
    assert FluxVT is SharedVT


def test_apps_keep_their_own_wire_versions() -> None:
    """The per-app wire/code versions stay in each app's wire_version.py."""
    from zcu_tools.gui.app.fluxdep.services.remote import wire_version as flux_wv
    from zcu_tools.gui.app.main.services.remote import wire_version as main_wv

    # Distinct contracts: measure-gui has evolved its wire; fluxdep starts at 1.
    assert main_wv.WIRE_VERSION >= 1
    assert flux_wv.WIRE_VERSION == 1
    assert flux_wv.GUI_VERSION == 1


# Each mcp server is launched as a SCRIPT (``uv run .../mcp/<app>/server.py`` per
# .mcp.json), so it has no parent package — a relative import would die at launch
# with "attempted relative import with no known parent package", which only
# surfaces on an MCP reconnect (not in the test suite, which imports it as a
# module). Guard the absolute-import invariant statically so a future
# import-rewrite cannot regress it silently. (The servers were consolidated under
# zcu_tools/mcp/<app>/server.py — one sub-package per server.)
_MCP_SERVERS = [
    "lib/zcu_tools/mcp/measure/server.py",
    "lib/zcu_tools/mcp/fluxdep/server.py",
    "lib/zcu_tools/mcp/dispersive/server.py",
    "lib/zcu_tools/mcp/agent_memory/server.py",
]


@pytest.mark.parametrize("rel_path", _MCP_SERVERS)
def test_mcp_server_has_no_relative_imports(rel_path: str) -> None:
    import ast

    repo_root = Path(__file__).resolve().parents[2]
    tree = ast.parse((repo_root / rel_path).read_text(encoding="utf-8"))
    offenders = [
        f"line {node.lineno}: from {'.' * node.level}{node.module or ''} import ..."
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level > 0
    ]
    assert not offenders, (
        f"{rel_path} is launched as a script and must use absolute imports; "
        f"found relative: {offenders}"
    )
