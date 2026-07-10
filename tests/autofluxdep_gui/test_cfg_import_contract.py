"""Import ownership contract for the autoflux-local cfg barrel."""

from __future__ import annotations

import ast
from pathlib import Path

import zcu_tools.gui.app.autofluxdep.cfg as autoflux_cfg
import zcu_tools.gui.cfg as shared_cfg

_AUTOFLUX_CFG_BARREL = "zcu_tools.gui.app.autofluxdep.cfg"


def test_autoflux_cfg_package_exports_only_app_owned_names() -> None:
    generic_names = frozenset(shared_cfg.__all__)

    assert not generic_names.intersection(autoflux_cfg.__all__)
    assert not generic_names.intersection(vars(autoflux_cfg))


def test_source_imports_shared_cfg_names_from_shared_owner() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    generic_names = frozenset(shared_cfg.__all__)
    offenders: list[str] = []

    for source_root in (repo_root / "lib", repo_root / "tests"):
        for path in sorted(source_root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                module = node.module or ""
                if not (
                    module == _AUTOFLUX_CFG_BARREL
                    or module.startswith(f"{_AUTOFLUX_CFG_BARREL}.")
                ):
                    continue
                for alias in node.names:
                    if alias.name == "*" or alias.name in generic_names:
                        relative_path = path.relative_to(repo_root)
                        offenders.append(f"{relative_path}:{node.lineno}:{alias.name}")

    assert not offenders, "\n".join(offenders)
