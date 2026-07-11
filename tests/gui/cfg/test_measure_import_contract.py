"""Import ownership contract for the measure adapter facade."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import zcu_tools.experiment.v2_gui.adapters._support.defaults as role_defaults
import zcu_tools.gui.app.main.adapter as measure_adapter
import zcu_tools.gui.cfg as shared_cfg

_MEASURE_GENERIC_FACADES = (
    "zcu_tools.gui.app.main.adapter",
    "zcu_tools.experiment.v2_gui.adapters._support.defaults",
)


def test_measure_adapter_package_does_not_export_shared_cfg_names() -> None:
    generic_names = frozenset(shared_cfg.__all__)
    facade_all = frozenset(getattr(measure_adapter, "__all__", ()))

    assert not generic_names.intersection(facade_all)
    assert not generic_names.intersection(vars(measure_adapter))
    assert not generic_names.intersection(role_defaults.__all__)
    assert not generic_names.intersection(vars(role_defaults))


def test_source_imports_shared_cfg_names_from_shared_owner() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    generic_names = frozenset(shared_cfg.__all__)
    offenders: list[str] = []

    for source_root in (repo_root / "lib", repo_root / "tests"):
        for path in sorted(source_root.rglob("*.py")):
            tree = ast.parse(
                path.read_text(encoding="utf-8"),
                filename=str(path),
            )
            package = _source_package(repo_root, path)
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                module = _resolve_import_module(node, package)
                if not any(
                    module == facade or module.startswith(f"{facade}.")
                    for facade in _MEASURE_GENERIC_FACADES
                ):
                    continue
                for alias in node.names:
                    if alias.name == "*" or alias.name in generic_names:
                        relative_path = path.relative_to(repo_root)
                        offenders.append(
                            f"{relative_path}:{node.lineno}:{module}:{alias.name}"
                        )

    assert not offenders, "\n".join(offenders)


def _source_package(repo_root: Path, path: Path) -> str:
    module_parts = path.relative_to(repo_root).with_suffix("").parts
    if module_parts[0] == "lib":
        module_parts = module_parts[1:]
    return ".".join(module_parts[:-1])


def _resolve_import_module(node: ast.ImportFrom, package: str) -> str:
    module = node.module or ""
    if node.level == 0:
        return module
    return importlib.util.resolve_name("." * node.level + module, package)
