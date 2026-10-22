"""Shared fake kind, roots and declarations, activated by converter modules only."""

from collections.abc import Generator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from zcu_tools.datafile import DataVariable, JsonObject
from zcu_tools.resources.entry import ComponentSchema, component_registry

from migrations.legacy_labber import LegacyAxisSchema, LegacyVariableSchema
from migrations.storage_mapping import build_mapping
from migrations.storage_migration import (
    KeyRule,
    MigrationMapping,
    MigrationRequest,
    ModuleRule,
)
from tests.migrations.storage_migration.fakes import SyntheticProbe
from tests.migrations.storage_migration.params_support import AnalysisQubit
from tests.resources.entry.fakes import registry_state


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
                LegacyVariableSchema(
                    variable=DataVariable("signal"),
                    axes=(
                        LegacyAxisSchema(name="x", unit="s", dtype=np.dtype("float64")),
                    ),
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
        result_root=tmp_path / "repo" / "result",
        database_root=tmp_path / "repo" / "Database",
        results_root=tmp_path / "repo" / "results",
        repo_root=tmp_path / "repo",
        backup_parent=tmp_path,
        source_chip="chip",
        source_qubit="qubit",
        name="destination",
        part="all",
    )
    (request.result_root / "chip" / "qubit").mkdir(parents=True)
    (request.database_root / "chip" / "qubit").mkdir(parents=True)
    return request


@pytest.fixture(scope="module")
def analysis_kind() -> Generator[None]:
    before = registry_state()
    models = {
        "qubit/migration_analysis": AnalysisQubit,
        "resonator/migration_analysis": AnalysisQubit,
        "amplifier/migration_analysis": AnalysisQubit,
        "qubit/migration_no_ext": ComponentSchema,
    }
    registered: list[str] = []
    try:
        for kind, model in models.items():
            component_registry.register(kind, model)
            registered.append(kind)
        yield
    finally:
        for kind in reversed(registered):
            component_registry.unregister(kind)
        assert registry_state() == before


@pytest.fixture
def analysis_mapping() -> MigrationMapping:
    return MigrationMapping(
        mapping_version="1.0",
        components={
            "Qlegacy": {"kind": "qubit/migration_analysis", "ext": {"keep": 7}}
        },
        roles={"qubit": "Qlegacy"},
        rules=(
            KeyRule(
                old_key="energy",
                target_path="Qlegacy.EJ",
                action="value",
                reason="Explicit",
            ),
            KeyRule(
                old_key="unit",
                target_path="general.ext.flux_unit",
                action="value",
                reason="Explicit",
            ),
        ),
        data_schemas={},
    )


@pytest.fixture
def legacy_fit() -> JsonObject:
    return {
        "params": {"EJ": 6.27, "EC": 1.41, "EL": 0.8},
        "flux_half": 0.2,
        "flux_int": -0.3,
        "flux_period": 1.0,
        "timestamp": "2020-07-07T16:30:00+08:00",
        "plot_transitions": {"0.1": [1, 2]},
    }


@pytest.fixture
def metadata_mapping(analysis_mapping: MigrationMapping) -> MigrationMapping:
    concrete = build_mapping(
        qubit_kind="qubit/fluxonium", data_schemas={}, native_tags={}
    )
    return replace(
        concrete,
        components={
            "Q1": {"kind": "qubit/migration_analysis"},
            "R1": {"kind": "resonator/migration_analysis"},
            "J1": {"kind": "amplifier/migration_analysis"},
        },
        roles={"qubit": "Q1"},
    )


@pytest.fixture
def analysis_sections() -> JsonObject:
    return {
        "dispersive": {
            "g": None,
            "bare_rf": 5.1,
            "timestamp": "unproven",
            "future": [1, None],
        },
        "fluxdep_fit": {"plot_transitions": {"r_f": 5.2, "0.1": [1, 2]}},
        "t1_curve_fit": {
            "params": {"Q_cap": None, "Temp": 0.04},
            "stderr": {"Q_cap": None},
            "bounds": {"Q_cap": [1, None]},
            "free": ["Q_cap"],
            "fixed": ["Temp"],
            "cost": None,
            "timestamp": "2020-07-07T16:30:00+08:00",
        },
    }


@pytest.fixture
def library_mapping(mapping: MigrationMapping) -> MigrationMapping:
    return replace(
        mapping,
        module_rules=(
            ModuleRule(
                old_name="same",
                target_path="C1.pulses.pi",
                reference_path="C1.module.chosen",
                reason="Explicit pulse",
            ),
            ModuleRule(
                old_name="readout",
                target_path="C1.readout.main",
                reference_path=None,
                reason="Explicit readout",
            ),
            *(
                ModuleRule(
                    old_name=old,
                    target_path=f"C1.waveforms.{new}",
                    reference_path=None,
                    reason="Explicit waveform",
                    kind="waveform",
                )
                for old, new in (("same", "base"), ("ramp", "ramp"), ("flat", "flat"))
            ),
        ),
    )
