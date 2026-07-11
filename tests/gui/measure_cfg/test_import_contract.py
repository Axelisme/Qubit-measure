"""Import-boundary checks for the measure-domain cfg catalog."""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path


def test_measure_cfg_import_is_app_runtime_and_qt_clean() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(repo_root / "lib")!r})
        import zcu_tools.gui.measure_cfg  # noqa: F401

        forbidden = (
            "zcu_tools.gui.app",
            "zcu_tools.gui.session",
            "zcu_tools.experiment",
            "zcu_tools.meta_tool",
            "zcu_tools.program",
            "zcu_tools.device",
            "zcu_tools.notebook",
            "qtpy",
            "PyQt",
            "PySide",
        )
        leaked = sorted(name for name in sys.modules if name.startswith(forbidden))
        assert not leaked, f"importing gui.measure_cfg leaked: {{leaked}}"
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


def test_measure_cfg_source_has_no_forbidden_runtime_imports() -> None:
    catalog_dir = (
        Path(__file__).resolve().parents[3]
        / "lib"
        / "zcu_tools"
        / "gui"
        / "measure_cfg"
    )
    forbidden = (
        "zcu_tools.gui.app",
        "zcu_tools.gui.session",
        "zcu_tools.experiment",
        "zcu_tools.meta_tool",
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


def test_gui_cfg_import_does_not_load_measure_cfg() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(repo_root / "lib")!r})
        import zcu_tools.gui.cfg  # noqa: F401
        assert "zcu_tools.gui.measure_cfg" not in sys.modules
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
