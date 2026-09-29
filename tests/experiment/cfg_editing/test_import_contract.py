"""Import-boundary checks for the measure-domain cfg catalog."""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path


def test_cfg_editing_import_is_app_runtime_and_qt_clean() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(repo_root / "lib")!r})
        import zcu_tools.experiment.cfg_editing  # noqa: F401

        # The experiment root eagerly imports base and its device/datafile dependencies (D6, documentation-structure-refresh).
        forbidden = (
            "zcu_tools.gui.app",
            "zcu_tools.gui.session",
            "zcu_tools.experiment.v2",
            "zcu_tools.resources",
            "zcu_tools.program",
            "zcu_tools.notebook",
            "qtpy",
            "PyQt",
            "PySide",
        )
        leaked = sorted(name for name in sys.modules if name.startswith(forbidden))
        assert not leaked, f"importing experiment.cfg_editing leaked: {{leaked}}"
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


def test_cfg_editing_source_has_no_forbidden_runtime_imports() -> None:
    catalog_dir = (
        Path(__file__).resolve().parents[3]
        / "lib"
        / "zcu_tools"
        / "experiment"
        / "cfg_editing"
    )
    # The experiment root eagerly imports base and its device/datafile dependencies (D6, documentation-structure-refresh).
    forbidden = (
        "zcu_tools.gui.app",
        "zcu_tools.gui.session",
        "zcu_tools.experiment.v2",
        "zcu_tools.resources",
        "zcu_tools.program",
        "zcu_tools.device",
        "zcu_tools.notebook",
        "qtpy",
        "PyQt",
        "PySide",
    )
    offenders: list[str] = []
    for path in sorted(catalog_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                modules.append(node.module)
            elif isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            for module in modules:
                if module.startswith(forbidden):
                    offenders.append(
                        f"{path.name}:{getattr(node, 'lineno', 0)}:{module}"
                    )
    assert not offenders, "\n".join(offenders)


def test_gui_cfg_import_does_not_load_cfg_editing() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(repo_root / "lib")!r})
        import zcu_tools.gui.cfg  # noqa: F401
        assert "zcu_tools.experiment.cfg_editing" not in sys.modules
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
