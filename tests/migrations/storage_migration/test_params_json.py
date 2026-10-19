"""Legacy analysis handoff conversion through migrate_storage and ResultEntry.

Remove with the offline migrations package once legacy entries are converted.
"""

import json
from collections.abc import Generator
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ConfigDict, Field
from zcu_tools.datafile import JsonObject
from zcu_tools.resources.document_store import YamlMap
from zcu_tools.resources.entry import (
    ComponentSchema,
    ResultEntry,
    component_registry,
    document_value,
    snapshot,
)

from migrations.storage_migration import (
    KeyRule,
    MigrationMapping,
    MigrationReport,
    MigrationRequest,
    migrate_storage,
)
from tests.migrations.storage_migration.fakes import (
    noop_validation,
    validate_cfg,
    write_meta,
)
from tests.resources.entry.fakes import registry_state


class AnalysisQubit(ComponentSchema):
    """Synthetic qubit with nullable analysis fields and no lab dependencies."""

    model_config = ConfigDict(extra="forbid")
    EJ: float | None = None
    EC: float | None = None
    EL: float | None = None
    flux_half: float | None = None
    flux_int: float | None = None
    flux_period: float | None = None
    ext: YamlMap = Field(default_factory=dict)


class ChargeOnlyQubit(ComponentSchema):
    """Synthetic qubit kind without formal Josephson or inductive energy fields."""

    model_config = ConfigDict(extra="forbid")
    EC: float | None = None
    ext: YamlMap = Field(default_factory=dict)


@pytest.fixture(scope="module", autouse=True)
def analysis_kind() -> Generator[None]:
    before = registry_state()
    component_registry.register("qubit/migration_analysis", AnalysisQubit)
    component_registry.register("qubit/migration_charge_only", ChargeOnlyQubit)
    try:
        yield
    finally:
        component_registry.unregister("qubit/migration_charge_only")
        component_registry.unregister("qubit/migration_analysis")
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


def write_params(request: MigrationRequest, values: JsonObject) -> Path:
    """Write one synthetic legacy handoff in the fixture source tree."""
    path = request.result_root / "chip" / "qubit" / "params.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    return path


def migrate(request: MigrationRequest, mapping: MigrationMapping) -> MigrationReport:
    """Run only parameter migration through its public composition seam."""
    return migrate_storage(
        replace(request, part="parameters"),
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )


def open_entry(request: MigrationRequest) -> ResultEntry:
    """Open the synthetic destination with explicit roots."""
    return ResultEntry.open(
        request.name,
        result_root=request.results_root,
        database_root=request.database_root,
    )


def test_fit_fields_and_lossless_sections_reach_named_qubit(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
) -> None:
    fit: JsonObject = {
        "params": {"EJ": 6.27, "EC": 1.41, "EL": 0.8},
        "flux_half": 0.2,
        "timestamp": "historical",
        "plot_transitions": {"0.1": [1, 2]},
    }
    dispersive: JsonObject = {"g": 0.03, "timestamp": "original", "future": {"x": True}}
    source = write_params(
        request_data,
        {"fluxdep_fit": fit, "dispersive": dispersive, "project": {"chip": "old"}},
    )
    before = source.read_bytes()
    report = migrate(request_data, analysis_mapping)
    doc = snapshot(open_entry(request_data).setup)
    assert doc.components["Qlegacy"].model_dump()["EJ"] == 6.27
    assert document_value(doc, "Qlegacy.EC") == 1.41
    assert document_value(doc, "Qlegacy.EL") == 0.8
    assert document_value(doc, "Qlegacy.ext.keep") == 7
    assert document_value(doc, "Qlegacy.ext.legacy_params.dispersive") == dispersive
    assert document_value(doc, "Qlegacy.ext.legacy_params.fluxdep_fit") == fit
    assert doc.components["Qlegacy"].model_dump()["flux_half"] is None
    assert any(
        item.source == source and item.location == "fluxdep_fit.flux_half"
        for item in report.pending
    )
    assert any(
        item.old_key == "dispersive"
        and item.new_path == "Qlegacy.ext.legacy_params.dispersive"
        and item.action == "pending"
        for item in report.key_mappings
    )
    assert source.read_bytes() == before


def test_meta_values_win_per_context_and_params_fill_other_fields(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
) -> None:
    root = write_meta(request_data, "", {"energy": 9.0})
    point = write_meta(request_data, "bias", {"energy": 11.0})
    source = write_params(
        request_data, {"fluxdep_fit": {"params": {"EJ": 6.27, "EC": 1.41}}}
    )
    report = migrate(request_data, analysis_mapping)
    entry = open_entry(request_data)
    assert snapshot(entry.setup).components["Qlegacy"].model_dump()["EC"] == 1.41
    assert document_value(snapshot(entry.setup), "Qlegacy.EJ") == 9.0
    assert document_value(snapshot(entry.use_point("bias")), "Qlegacy.EJ") == 11.0
    assert document_value(snapshot(entry.use_point("bias")), "Qlegacy.EC") == 1.41
    conflicts = [
        item
        for item in report.pending
        if item.source == source and item.location == "fluxdep_fit.params.EJ"
    ]
    assert conflicts
    reasons = " ".join(item.reason for item in conflicts)
    assert str(root) in reasons and str(point) in reasons
    assert "9.0" in reasons and "11.0" in reasons and "6.27" in reasons


@pytest.mark.parametrize(
    "components",
    [
        {},
        {
            "Qlegacy": {"kind": "qubit/migration_analysis"},
            "Qother": {"kind": "qubit/migration_analysis"},
        },
    ],
)
def test_no_unique_qubit_keeps_whole_handoff_pending(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
    components: dict[str, JsonObject],
) -> None:
    source = write_params(request_data, {"fluxdep_fit": {"params": {"EJ": 6.27}}})
    report = migrate(
        request_data,
        replace(analysis_mapping, components=components, rules=(), roles={}),
    )
    assert any(
        item.source == source
        and item.location == "params.json"
        and "unique" in item.reason
        for item in report.pending
    )
    assert all(
        snapshot(open_entry(request_data).setup).components[name].model_dump()["EJ"]
        is None
        for name in components
    )


@pytest.mark.parametrize("root_unit,fit_unit", [("A", None), (None, "mA"), ("V", "V")])
def test_explicit_flux_unit_is_kept_in_each_context(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
    root_unit: str | None,
    fit_unit: str | None,
) -> None:
    write_meta(request_data, "bias", {})
    fit: JsonObject = {
        "params": {"EJ": 6.27},
        "flux_half": 0.2,
        "flux_int": -0.3,
        "flux_period": 1.0,
    }
    values: JsonObject = {"fluxdep_fit": fit}
    if root_unit is not None:
        values["flux_unit"] = root_unit
    if fit_unit is not None:
        fit["flux_unit"] = fit_unit
    write_params(request_data, values)
    migrate(request_data, analysis_mapping)
    entry = open_entry(request_data)
    for view in (entry.setup, entry.use_point("bias")):
        doc = snapshot(view)
        assert doc.components["Qlegacy"].model_dump()["flux_half"] == 0.2
        assert document_value(doc, "Qlegacy.flux_int") == -0.3
        assert document_value(doc, "Qlegacy.flux_period") == 1.0
        assert document_value(doc, "general.ext.flux_unit") == (root_unit or fit_unit)


def test_disagreeing_source_units_leave_flux_pending(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
) -> None:
    source = write_params(
        request_data,
        {"flux_unit": "A", "fluxdep_fit": {"flux_unit": "V", "flux_half": 0.2}},
    )
    report = migrate(request_data, analysis_mapping)
    assert any(
        item.source == source and item.location == "fluxdep_fit.flux_half"
        for item in report.pending
    )
    assert (
        snapshot(open_entry(request_data).setup)
        .components["Qlegacy"]
        .model_dump()["flux_half"]
        is None
    )


def test_completed_resume_preserves_user_edits(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
) -> None:
    write_meta(request_data, "bias", {})
    write_params(request_data, {"fluxdep_fit": {"params": {"EJ": 6.27}}})
    migrate(request_data, analysis_mapping)
    entry = open_entry(request_data)
    assert snapshot(entry.setup).components["Qlegacy"].model_dump()["EJ"] == 6.27
    entry.setup.Qlegacy.EJ = 12.0
    entry.use_point("bias").Qlegacy.EJ = 13.0
    paths = [
        entry.result_path / "setup.yaml",
        entry.result_path / "points/bias/point.yaml",
    ]
    before = [path.read_bytes() for path in paths]
    migrate(replace(request_data, resume=True), analysis_mapping)
    assert [path.read_bytes() for path in paths] == before


def test_dry_run_reports_conversion_without_destination_writes(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
) -> None:
    source = write_params(
        request_data,
        {
            "fluxdep_fit": {"params": {"EJ": 6.27}},
            "t1_curve_fit": {"params": {"Q_cap": 1}},
        },
    )
    dry = migrate(replace(request_data, dry_run=True), analysis_mapping)
    assert any(
        item.old_file == source
        and item.old_key == "fluxdep_fit.params.EJ"
        and item.action == "value"
        for item in dry.key_mappings
    )
    assert not (request_data.results_root / request_data.name).exists()
    actual = migrate(request_data, analysis_mapping)
    assert actual.key_mappings == dry.key_mappings
    assert actual.pending == dry.pending


def test_point_unit_conflict_does_not_reinterpret_params_flux(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
) -> None:
    meta = write_meta(request_data, "bias", {"unit": "V"})
    source = write_params(
        request_data, {"flux_unit": "A", "fluxdep_fit": {"flux_half": 0.2}}
    )
    report = migrate(request_data, analysis_mapping)
    entry = open_entry(request_data)
    assert snapshot(entry.setup).components["Qlegacy"].model_dump()["flux_half"] == 0.2
    point = snapshot(entry.use_point("bias"))
    assert document_value(point, "general.ext.flux_unit") == "V"
    assert point.components["Qlegacy"].model_dump()["flux_half"] is None
    assert any(
        item.source == source
        and str(meta) in item.reason
        and "A" in item.reason
        and "V" in item.reason
        for item in report.pending
    )


@pytest.mark.parametrize("value", ["${md.energy}", True, "6.27", None])
def test_non_numeric_fit_fields_remain_located_pending(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
    value: str | bool | None,
) -> None:
    source = write_params(request_data, {"fluxdep_fit": {"params": {"EJ": value}}})
    report = migrate(request_data, analysis_mapping)
    assert any(
        item.source == source and item.location == "fluxdep_fit.params.EJ"
        for item in report.pending
    )
    assert (
        snapshot(open_entry(request_data).setup)
        .components["Qlegacy"]
        .model_dump()["EJ"]
        is None
    )


def test_unsupported_energy_fields_are_preserved_without_extending_schema(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
) -> None:
    fit: JsonObject = {"params": {"EJ": 6.27, "EC": 1.41, "EL": 0.8}}
    source = write_params(request_data, {"fluxdep_fit": fit})
    mapping = replace(
        analysis_mapping,
        components={"Qlegacy": {"kind": "qubit/migration_charge_only"}},
    )
    report = migrate(request_data, mapping)
    doc = snapshot(open_entry(request_data).setup)
    assert document_value(doc, "Qlegacy.EC") == 1.41
    assert document_value(doc, "Qlegacy.ext.legacy_params.fluxdep_fit") == fit
    pending = {item.location for item in report.pending if item.source == source}
    assert {"fluxdep_fit.params.EJ", "fluxdep_fit.params.EL"} <= pending
