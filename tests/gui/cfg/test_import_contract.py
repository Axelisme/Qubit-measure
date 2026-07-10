"""Import-boundary checks for the shared GUI cfg core."""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path


def test_gui_cfg_import_is_app_experiment_meta_and_qt_clean() -> None:
    """The shared cfg package must remain usable without app runtimes or Qt."""
    repo_root = Path(__file__).resolve().parents[3]
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(repo_root / "lib")!r})
        import zcu_tools.gui.cfg  # noqa: F401

        qt_roots = {"qtpy", "PyQt6", "PyQt5", "PySide6", "PySide2"}
        leaked = sorted(
            name
            for name in sys.modules
            if name.split(".")[0] in qt_roots
            or name == "zcu_tools.gui.app"
            or name.startswith("zcu_tools.gui.app.")
            or name == "zcu_tools.experiment"
            or name.startswith("zcu_tools.experiment.")
            or name == "zcu_tools.meta_tool"
            or name.startswith("zcu_tools.meta_tool.")
            or name == "zcu_tools.notebook"
            or name.startswith("zcu_tools.notebook.")
            or name == "zcu_tools.device"
            or name.startswith("zcu_tools.device.")
        )
        assert not leaked, f"importing zcu_tools.gui.cfg leaked: {{leaked}}"
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


def test_gui_cfg_source_has_no_forbidden_runtime_imports() -> None:
    cfg_dir = Path(__file__).resolve().parents[3] / "lib" / "zcu_tools" / "gui" / "cfg"
    forbidden_prefixes = (
        "zcu_tools.gui.app",
        "zcu_tools.experiment",
        "zcu_tools.meta_tool",
        "zcu_tools.notebook",
        "zcu_tools.device",
        "qtpy",
        "PyQt6",
        "PyQt5",
        "PySide6",
        "PySide2",
    )
    offenders: list[str] = []
    for path in sorted(cfg_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                modules.append(node.module)
            elif isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            for module in modules:
                if module.startswith(forbidden_prefixes):
                    offenders.append(
                        f"{path.relative_to(cfg_dir)}:{getattr(node, 'lineno', 0)}: "
                        f"forbidden import {module}"
                    )
    assert not offenders, "\n".join(offenders)


def test_gui_cfg_lowering_has_no_environment_lookup_or_hidden_resolver_state() -> None:
    lowering_path = (
        Path(__file__).resolve().parents[3]
        / "lib"
        / "zcu_tools"
        / "gui"
        / "cfg"
        / "lowering.py"
    )
    tree = ast.parse(lowering_path.read_text(encoding="utf-8"))

    os_aliases = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == "os"
    }
    environment_lookups = [
        f"{node.value.id}.{node.attr}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in os_aliases
        and node.attr in {"getenv", "environ"}
    ]
    environment_lookups.extend(
        f"os.{alias.name}"
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "os"
        for alias in node.names
        if alias.name in {"getenv", "environ"}
    )

    hidden_state: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        else:
            continue
        if isinstance(value, (ast.Subscript, ast.BinOp)):
            continue
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            normalized = target.id.lower()
            if normalized.endswith(
                ("_resolver", "_resolvers", "_registry", "_registries")
            ):
                hidden_state.append(target.id)

    assert not environment_lookups, environment_lookups
    assert not hidden_state, hidden_state
