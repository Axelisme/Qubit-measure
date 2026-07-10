"""Catalog validation and experiment-package dependency architecture."""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, cast

import pytest
from zcu_tools.gui.app.autofluxdep.cfg import RunCfgSnapshot
from zcu_tools.gui.app.autofluxdep.experiments.catalog import (
    CATALOG,
    ExperimentCatalog,
    builders,
    create_placement,
    names,
)
from zcu_tools.gui.app.autofluxdep.nodes.builder import Builder, Node, RunEnv
from zcu_tools.gui.app.autofluxdep.orchestrator import Orchestrator

_EXPECTED_NAMES = (
    "qubit_freq",
    "lenrabi",
    "ro_optimize",
    "t1",
    "t2ramsey",
    "t2echo",
    "mist",
)
_EXPERIMENT_PREFIX = "zcu_tools.gui.app.autofluxdep.experiments."


def _catalog_builder(
    name: str,
    *,
    module_stem: str | None = None,
    provides: tuple[str, ...] = (),
) -> Builder:
    def build_node(self: Builder, env: RunEnv) -> Node:
        del self, env
        raise NotImplementedError

    builder_type = type(
        f"{name.title()}Builder",
        (Builder,),
        {
            "__module__": f"test_catalog.{module_stem or name or 'empty'}",
            "name": name,
            "provides": provides,
            "build_node": build_node,
        },
    )
    return builder_type()


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    source_root = Path(__file__).parents[3] / "lib"
    module_parts = path.relative_to(source_root).with_suffix("").parts
    if module_parts[-1] == "__init__":
        module_parts = module_parts[:-1]
        package = ".".join(module_parts)
    else:
        package = ".".join(module_parts[:-1])

    return _canonical_imports(tree, package=package)


def _canonical_imports(tree: ast.AST, *, package: str) -> tuple[str, ...]:
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                relative_name = "." * node.level + (node.module or "")
                imported_from = importlib.util.resolve_name(relative_name, package)
            elif node.module is not None:
                imported_from = node.module
            else:  # pragma: no cover - absolute ``from`` always has a module
                continue
            imported.append(imported_from)
            imported.extend(
                f"{imported_from}.{alias.name}"
                for alias in node.names
                if alias.name != "*"
            )
    return tuple(imported)


def test_import_analyzer_canonicalizes_relative_imports() -> None:
    tree = ast.parse(
        "from . import qubit_freq\n"
        "from .t2echo import EXPERIMENT\n"
        "from ..experiments import t1\n"
    )

    imported = _canonical_imports(
        tree,
        package="zcu_tools.gui.app.autofluxdep.experiments",
    )

    assert "zcu_tools.gui.app.autofluxdep.experiments.qubit_freq" in imported
    assert "zcu_tools.gui.app.autofluxdep.experiments.t2echo" in imported
    assert "zcu_tools.gui.app.autofluxdep.experiments.t1" in imported


def test_catalog_is_explicit_ordered_and_immutable() -> None:
    assert names() == _EXPECTED_NAMES
    assert tuple(builder.name for builder in builders()) == _EXPECTED_NAMES
    with pytest.raises(FrozenInstanceError):
        CATALOG._builders = ()  # type: ignore[misc]


def test_catalog_rejects_wrong_builder_type() -> None:
    with pytest.raises(TypeError, match="Builder instances"):
        ExperimentCatalog(cast("Any", (object(),)))


def test_catalog_rejects_empty_and_duplicate_names() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ExperimentCatalog((_catalog_builder(""),))
    with pytest.raises(ValueError, match="duplicate experiment name"):
        ExperimentCatalog((builders()[0], builders()[0]))


def test_catalog_rejects_module_stem_name_mismatch() -> None:
    with pytest.raises(ValueError, match="module stem"):
        ExperimentCatalog((_catalog_builder("declared", module_stem="actual"),))


def test_catalog_rejects_duplicate_declaration_entries() -> None:
    with pytest.raises(ValueError, match="duplicate provides declaration"):
        ExperimentCatalog((_catalog_builder("duplicate", provides=("x", "x")),))


def test_unknown_placement_preserves_key_error() -> None:
    with pytest.raises(KeyError):
        create_placement("not_registered")


def test_each_concrete_experiment_file_is_registered_exactly_once() -> None:
    package_dir = (
        Path(__file__).parents[3] / "lib/zcu_tools/gui/app/autofluxdep/experiments"
    )
    concrete = {
        path.stem
        for path in package_dir.glob("*.py")
        if path.stem not in {"__init__", "catalog"}
    }
    registered = tuple(
        builder.__class__.__module__.rsplit(".", 1)[-1] for builder in builders()
    )

    assert set(registered) == concrete
    assert len(registered) == len(set(registered))


def test_support_and_nodes_do_not_import_concrete_experiments() -> None:
    package_dir = Path(__file__).parents[3] / "lib/zcu_tools/gui/app/autofluxdep"
    concrete_modules = {_EXPERIMENT_PREFIX + name for name in _EXPECTED_NAMES}
    guarded_files = (
        *package_dir.joinpath("experiments/_support").rglob("*.py"),
        *package_dir.joinpath("nodes").rglob("*.py"),
    )

    for path in guarded_files:
        assert concrete_modules.isdisjoint(_imports(path)), path


def test_concrete_experiments_do_not_import_one_another() -> None:
    package_dir = (
        Path(__file__).parents[3] / "lib/zcu_tools/gui/app/autofluxdep/experiments"
    )
    concrete_modules = {_EXPERIMENT_PREFIX + name for name in _EXPECTED_NAMES}
    for name in _EXPECTED_NAMES:
        path = package_dir / f"{name}.py"
        assert concrete_modules.isdisjoint(_imports(path)), path


def test_support_import_does_not_load_catalog_or_concrete_experiments() -> None:
    forbidden = (
        "zcu_tools.gui.app.autofluxdep.experiments.catalog",
        *(_EXPERIMENT_PREFIX + name for name in _EXPECTED_NAMES),
    )
    probe = f"""
import sys
import zcu_tools.gui.app.autofluxdep.experiments._support.result

forbidden = {forbidden!r}
loaded = tuple(name for name in forbidden if name in sys.modules)
if loaded:
    raise SystemExit(f"unexpected experiment imports: {{loaded!r}}")
"""

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_predictor_is_not_user_placeable() -> None:
    assert "predictor" not in names()
    assert all(builder.name != "predictor" for builder in builders())


def test_catalog_does_not_reorder_user_workflow() -> None:
    requested = ("mist", "qubit_freq", "t1")
    providers = [create_placement(name) for name in requested]
    snapshots = {
        provider.name: RunCfgSnapshot(
            base_cfg={},
            override_plan=provider.builder.override_plan(provider.schema),
            knobs={},
        )
        for provider in providers
    }

    orchestrator = Orchestrator(providers, cfg_snapshots=snapshots)

    assert tuple(provider.type_name for provider in orchestrator.providers) == requested
