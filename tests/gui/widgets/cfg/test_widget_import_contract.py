"""Import and source boundaries for the shared Qt cfg widgets."""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path


def _widget_cfg_dir() -> Path:
    return (
        Path(__file__).resolve().parents[4]
        / "lib"
        / "zcu_tools"
        / "gui"
        / "widgets"
        / "cfg"
    )


def test_shared_cfg_widget_import_does_not_load_forbidden_layers() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    script = textwrap.dedent(
        """
        import sys
        import zcu_tools.gui.widgets.cfg  # noqa: F401

        forbidden = (
            "zcu_tools.gui.app",
            "zcu_tools.experiment",
            "zcu_tools.meta_tool",
            "zcu_tools.gui.event_bus",
            "zcu_tools.gui.session",
            "zcu_tools.device",
            "zcu_tools.notebook",
        )
        leaked = sorted(name for name in sys.modules if name.startswith(forbidden))
        assert not leaked, f"importing shared cfg widgets leaked: {leaked}"
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


def test_shared_cfg_widget_source_has_no_forbidden_imports() -> None:
    forbidden_prefixes = (
        "zcu_tools.gui.app",
        "zcu_tools.experiment",
        "zcu_tools.meta_tool",
        "zcu_tools.gui.event_bus",
        "zcu_tools.gui.session",
        "zcu_tools.device",
        "zcu_tools.notebook",
    )
    offenders: list[str] = []
    cfg_dir = _widget_cfg_dir()
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
                        f"{path.relative_to(cfg_dir)}:"
                        f"{getattr(node, 'lineno', 0)}: {module}"
                    )

    assert not offenders, "\n".join(offenders)


def test_registry_source_has_no_global_mapping_or_registration_decorator() -> None:
    registry_path = _widget_cfg_dir() / "registry.py"
    source = registry_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    banned_names = {"WIDGET_REGISTRY", "register_widget", "get_widget_cls"}
    assigned_module_dicts = [
        target.id
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name) and isinstance(node.value, ast.Dict)
    ]

    assert not banned_names.intersection(source.split())
    assert not assigned_module_dicts
    assert "@register" not in source
