"""Catalog validation and workflow placement behavior."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest
from zcu_tools.experiment.v2_gui.autofluxdep.catalog import (
    CATALOG,
    ExperimentCatalog,
    builders,
    create_placement,
    names,
)
from zcu_tools.gui.app.autofluxdep.cfg import RunCfgSnapshot
from zcu_tools.gui.app.autofluxdep.nodes.builder import Builder, Node, RunEnv
from zcu_tools.gui.app.autofluxdep.orchestrator import Orchestrator

from tests.gui.app.autofluxdep._helpers import make_run_context

_EXPECTED_NAMES = (
    "qubit_freq",
    "lenrabi",
    "ro_optimize",
    "t1",
    "t2ramsey",
    "t2echo",
    "mist",
)


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

    orchestrator = Orchestrator(
        providers,
        cfg_snapshots=snapshots,
        context=make_run_context(),
        device_snapshot={},
    )

    assert tuple(provider.type_name for provider in orchestrator.providers) == requested
