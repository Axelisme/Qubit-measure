"""Shared fake kind, roots and declarations, activated by converter modules only."""

from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
from zcu_tools.datafile import AxisSchema, DataVariable, VariableSchema
from zcu_tools.resources.entry import component_registry
from zcu_tools.resources.storage_migration import (
    KeyRule,
    MigrationMapping,
    MigrationRequest,
)

from tests.resources.entry.fakes import registry_state
from tests.resources.storage_migration.fakes import SyntheticProbe


@pytest.fixture(scope="module")
def registered_kind() -> Generator[None]:
    before = registry_state()
    component_registry.register("migration_test_probe", SyntheticProbe)
    expected = registry_state()
    try:
        yield
        assert registry_state() == expected, "migration module polluted registries"
    finally:
        component_registry.unregister("migration_test_probe")
        assert registry_state() == before


@pytest.fixture
def registry_guard(
    request: pytest.FixtureRequest, registered_kind: None
) -> Generator[None]:
    before = registry_state()
    yield
    assert registry_state() == before, f"registry polluter: {request.node.nodeid}"


@pytest.fixture
def mapping(registered_kind: None) -> MigrationMapping:
    return MigrationMapping(
        mapping_version="1.0",
        components={"C1": {"kind": "migration_test_probe", "frequency": 1.0}},
        rules=(
            KeyRule(
                old_key="old_frequency",
                target_path="C1.frequency",
                action="value",
                reason="Explicit mapping",
            ),
            KeyRule(
                old_key="old_error",
                target_path="C1.frequency",
                action="stderr",
                reason="Same working unit",
            ),
            KeyRule(
                old_key="obsolete",
                target_path=None,
                action="remove",
                reason="No longer used",
            ),
        ),
        roles={"probe": "C1"},
        data_schemas={
            ("synthetic_scan", "SyntheticCfg"): (
                VariableSchema(
                    variable=DataVariable("signal"),
                    axes=(AxisSchema(name="x", unit="s", dtype=np.dtype("float64")),),
                    signal_name="signal",
                    signal_unit="V",
                    signal_dtype=np.dtype("complex128"),
                ),
            )
        },
    )


@pytest.fixture
def request_data(tmp_path: Path) -> MigrationRequest:
    request = MigrationRequest(
        result_root=tmp_path / "result",
        database_root=tmp_path / "Database",
        results_root=tmp_path / "results",
        source_chip="chip",
        source_qubit="qubit",
        name="destination",
        part="all",
    )
    (request.result_root / "chip" / "qubit").mkdir(parents=True)
    (request.database_root / "chip" / "qubit").mkdir(parents=True)
    return request
