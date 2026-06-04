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
