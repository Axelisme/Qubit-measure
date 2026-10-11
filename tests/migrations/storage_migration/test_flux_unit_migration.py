"""Lab global flux migration keeps caller working units and source evidence."""

from dataclasses import replace
from pathlib import Path

import pytest
from zcu_tools.resources.entry import (
    ResultEntry,
    component_registry,
    document_value,
    snapshot,
)

from migrations.storage_migration import MigrationRequest, migrate_storage
from tests.migrations.storage_migration.fakes import (
    noop_validation,
    validate_cfg,
    write_meta,
)
from tests.resources.entry.fakes import registry_state, restore_registry
from zcu_lab.components import JpaSchema, QubitSchema, ResonatorSchema
from zcu_lab.storage_migration import build_mapping


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


@pytest.mark.parametrize("unit", ["A", "V"])
def test_lab_global_flux_mapping_keeps_working_units_in_general_extensions(
    request_data: MigrationRequest, unit: str
) -> None:
    before_registry = registry_state()
    try:
        component_registry.register("resonator", ResonatorSchema)
        component_registry.register("qubit/fluxonium", QubitSchema)
        component_registry.register("amplifier/jpa", JpaSchema)
        mapping = build_mapping(
            qubit_kind="qubit/fluxonium", data_schemas={}, native_tags={}
        )
        setup_source = write_meta(request_data, "", {"flux_unit": unit, "cur_A": 0.375})
        point_source = write_meta(
            request_data,
            "working",
            {
                "flux_unit": unit,
                "cur_A": -12.5,
                "flx_half": 0.5,
                "flx_period": 2.0,
                "flx_int": 1.25,
                "local_flux_unit": "V",
                "jpa_flux_unit": "A",
                "best_jpa_flux": 0.125,
            },
        )
        before = (setup_source.read_bytes(), point_source.read_bytes())
        report = migrate_storage(
            replace(request_data, part="parameters"),
            mapping=mapping,
            validate_cfg=validate_cfg,
            validate_native=noop_validation,
        )
        assert report.pending == ()
        entry = ResultEntry.open(
            request_data.name,
            result_root=request_data.results_root,
            database_root=request_data.database_root,
        )
        for view, value in ((entry.setup, 0.375), (entry.use_point("working"), -12.5)):
            assert snapshot(view).general.ext == {
                "flux_unit": unit,
                "flux_value": value,
            }
            assert document_value(snapshot(view), "general.ext.flux_value") == value
            source = view.meta("general.ext.flux_value")
            assert source is not None and source.source == "manual"
            assert source.stderr is None
        point = entry.use_point("working")
        assert point.Q1.flux_unit == "V"
        assert point.J1.flux_unit == "A"
        assert point.J1.flux == 0.125
        assert (point.Q1.flux_half, point.Q1.flux_period, point.Q1.flux_int) == (
            0.5,
            2.0,
            1.25,
        )
        assert (setup_source.read_bytes(), point_source.read_bytes()) == before
    finally:
        restore_registry(before_registry)
        assert registry_state() == before_registry
