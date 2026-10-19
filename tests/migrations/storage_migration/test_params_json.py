"""Legacy analysis handoff conversion through migrate_storage and ResultEntry.

Remove with the offline migrations package once legacy entries are converted.
"""

import json
from collections.abc import Generator
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ConfigDict, Field, TypeAdapter
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
    """Synthetic qubit with a meta-mapped energy and independent extension data."""

    model_config = ConfigDict(extra="forbid")
    EJ: float | None = None
    ext: YamlMap = Field(default_factory=dict)


@pytest.fixture(scope="module", autouse=True)
def analysis_kind() -> Generator[None]:
    before = registry_state()
    component_registry.register("qubit/migration_analysis", AnalysisQubit)
    try:
        yield
    finally:
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


def read_predictor(request: MigrationRequest) -> JsonObject:
    """Read the published predictor at the migration's public file boundary."""
    path = request.results_root / request.name / "predictors" / "Qlegacy.json"
    assert path.is_file(), "migrate_storage must publish the entry predictor"
    return TypeAdapter(JsonObject).validate_json(path.read_text(encoding="utf-8"))


def backup_path(request: MigrationRequest) -> Path:
    """Locate the explicit synthetic backup; never use production defaults."""
    if request.backup_parent is None:
        raise ValueError("Tests must supply a synthetic backup_parent")
    return request.backup_parent / f"{request.name}-backup"


def predictor_params(request: MigrationRequest) -> JsonObject:
    """Return the public predictor parameters as a typed JSON object."""
    return TypeAdapter(JsonObject).validate_python(read_predictor(request)["params"])


def test_lossless_sections_are_inherited_before_point_meta_overrides(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
) -> None:
    write_meta(request_data, "", {"energy": 9.0})
    write_meta(request_data, "bias", {"energy": 11.0})
    dispersive: JsonObject = {"g": 0.03, "timestamp": "original", "future": {"x": True}}
    source = write_params(
        request_data,
        {"dispersive": dispersive, "project": {"chip": "old"}},
    )
    before = source.read_bytes()
    report = migrate(request_data, analysis_mapping)
    entry = open_entry(request_data)
    for view, energy in ((entry.setup, 9.0), (entry.use_point("bias"), 11.0)):
        doc = snapshot(view)
        assert document_value(doc, "Qlegacy.EJ") == energy
        assert document_value(doc, "Qlegacy.ext.keep") == 7
        assert document_value(doc, "Qlegacy.ext.legacy_params.dispersive") == dispersive
        assert document_value(doc, "Qlegacy.ext.legacy_params.project") == {
            "chip": "old"
        }
    assert any(
        item.old_key == "dispersive"
        and item.new_path == "Qlegacy.ext.legacy_params.dispersive"
        and item.action == "pending"
        for item in report.key_mappings
    )
    assert source.read_bytes() == before


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


def test_completed_resume_preserves_user_edits(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
) -> None:
    write_meta(request_data, "bias", {})
    write_params(request_data, {"dispersive": {"g": 0.03}})
    migrate(request_data, analysis_mapping)
    entry = open_entry(request_data)
    entry.setup.Qlegacy.EJ = 12.0
    entry.use_point("bias").Qlegacy.EJ = 13.0
    paths = [
        entry.result_path / "setup.yaml",
        entry.result_path / "points/bias/point.yaml",
    ]
    before = [path.read_bytes() for path in paths]
    migrate(replace(request_data, resume=True), analysis_mapping)
    assert [path.read_bytes() for path in paths] == before


def test_dry_run_reports_preservation_without_destination_writes(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
) -> None:
    source = write_params(request_data, {"t1_curve_fit": {"params": {"Q_cap": 1}}})
    dry = migrate(replace(request_data, dry_run=True), analysis_mapping)
    assert any(
        item.old_file == source
        and item.old_key == "t1_curve_fit"
        and item.new_path == "Qlegacy.ext.legacy_params.t1_curve_fit"
        for item in dry.key_mappings
    )
    assert not (request_data.results_root / request_data.name).exists()
    actual = migrate(request_data, analysis_mapping)
    assert actual.key_mappings == dry.key_mappings
    assert actual.pending == dry.pending


@pytest.mark.parametrize("root_unit,fit_unit", [("A", None), (None, "mA"), ("V", "V")])
def test_legacy_fit_publishes_named_entry_predictor_with_native_values(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
    legacy_fit: JsonObject,
    root_unit: str | None,
    fit_unit: str | None,
) -> None:
    write_meta(request_data, "bias", {"energy": 11.0, "unit": "mV"})
    values: JsonObject = {"fluxdep_fit": legacy_fit}
    if root_unit is not None:
        values["flux_unit"] = root_unit
    if fit_unit is not None:
        legacy_fit["flux_unit"] = fit_unit
    write_params(request_data, values)
    migrate(request_data, analysis_mapping)
    assert read_predictor(request_data) == {
        "format": "zcu.predictor",
        "version": 1,
        "model": "fluxonium",
        "params": {
            "EJ": 6.27,
            "EC": 1.41,
            "EL": 0.8,
            "flux_half": 0.2,
            "flux_period": 1.0,
            "flux_bias": 0.0,
        },
        "flux_unit": root_unit or fit_unit,
        "source": {"run_id": None, "created_at": "2020-07-07T16:30:00+08:00"},
    }
    point = snapshot(open_entry(request_data).use_point("bias"))
    assert document_value(point, "Qlegacy.EJ") == 11.0
    assert document_value(point, "general.ext.flux_unit") == "mV"
    assert document_value(point, "Qlegacy.ext.legacy_params.fluxdep_fit") == legacy_fit


@pytest.mark.parametrize(
    "units", [{}, {"flux_unit": "A", "fit_unit": "V"}, {"flux_unit": "unknown"}]
)
def test_unknown_or_conflicting_units_publish_null_and_located_pending(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
    legacy_fit: JsonObject,
    units: dict[str, str],
) -> None:
    values: JsonObject = {"fluxdep_fit": legacy_fit}
    if "flux_unit" in units:
        values["flux_unit"] = units["flux_unit"]
    if "fit_unit" in units:
        legacy_fit["flux_unit"] = units["fit_unit"]
    source = write_params(request_data, values)
    report = migrate(request_data, analysis_mapping)
    assert read_predictor(request_data)["flux_unit"] is None
    assert predictor_params(request_data)["flux_half"] == 0.2
    assert any(
        item.source == source and item.location == "fluxdep_fit.flux_unit"
        for item in report.pending
    )


@pytest.mark.parametrize("timestamp", [None, "historical", "2020-07-07T16:30:00"])
def test_unproven_time_is_null_with_pending_at_verified_backup_original(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
    legacy_fit: JsonObject,
    timestamp: str | None,
) -> None:
    legacy_fit["timestamp"] = timestamp
    source = write_params(request_data, {"flux_unit": "A", "fluxdep_fit": legacy_fit})
    report = migrate(request_data, analysis_mapping)
    assert read_predictor(request_data)["source"] == {
        "run_id": None,
        "created_at": None,
    }
    backup_source = backup_path(request_data) / "result/params.json"
    pending = [
        item
        for item in report.pending
        if item.source == source and item.location == "fluxdep_fit.timestamp"
    ]
    assert pending
    assert str(backup_source) in pending[0].reason
    assert (
        TypeAdapter(JsonObject).validate_json(backup_source.read_text())["fluxdep_fit"]
        == legacy_fit
    )


@pytest.mark.parametrize("value", ["${md.energy}", True, "6.27", float("inf")])
def test_invalid_model_values_stay_pending_without_blocking_known_parameters(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
    legacy_fit: JsonObject,
    value: str | bool | float,
) -> None:
    legacy_fit["params"] = {"EJ": value, "EC": 1.41, "EL": None}
    source = write_params(request_data, {"flux_unit": "A", "fluxdep_fit": legacy_fit})
    report = migrate(request_data, analysis_mapping)
    params = predictor_params(request_data)
    assert params.get("EJ") is None
    assert params["EC"] == 1.41
    assert params.get("EL") is None
    assert any(
        item.source == source and item.location == "fluxdep_fit.params.EJ"
        for item in report.pending
    )


def test_partial_fit_keeps_unknowns_without_enforcing_physical_constraints(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
) -> None:
    write_params(
        request_data,
        {
            "flux_unit": "A",
            "fluxdep_fit": {"params": {"EC": -1.0, "EL": None}, "flux_period": 0.0},
        },
    )
    migrate(request_data, analysis_mapping)
    params = predictor_params(request_data)
    assert params.get("EJ") is None
    assert params["EC"] == -1.0
    assert params.get("EL") is None
    assert params.get("flux_half") is None
    assert params["flux_period"] == 0.0
    assert params["flux_bias"] == 0.0


@pytest.mark.parametrize("model", ["fluxonium", "transmon", "unrecognized"])
def test_undefined_explicit_model_contract_remains_located_pending(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
    legacy_fit: JsonObject,
    model: str,
) -> None:
    legacy_fit["model"] = model
    source = write_params(request_data, {"fluxdep_fit": legacy_fit})
    report = migrate(request_data, analysis_mapping)
    assert any(
        item.source == source
        and item.location == "fluxdep_fit.model"
        and "legacy" in item.reason.lower()
        for item in report.pending
    )
    assert (
        document_value(
            snapshot(open_entry(request_data).setup),
            "Qlegacy.ext.legacy_params.fluxdep_fit",
        )
        == legacy_fit
    )


def test_undefined_transmon_parameter_shape_stays_pending(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
) -> None:
    fit: JsonObject = {"params": {"EJmax": 12.0, "EC": 0.3, "d": 0.2}}
    source = write_params(request_data, {"fluxdep_fit": fit})
    report = migrate(request_data, analysis_mapping)
    assert any(
        item.source == source
        and item.location == "fluxdep_fit.params"
        and "legacy" in item.reason.lower()
        for item in report.pending
    )
    assert (
        document_value(
            snapshot(open_entry(request_data).setup),
            "Qlegacy.ext.legacy_params.fluxdep_fit",
        )
        == fit
    )


def test_dry_run_predictor_plan_matches_apply_and_completed_resume_keeps_edits(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
    legacy_fit: JsonObject,
) -> None:
    source = write_params(request_data, {"flux_unit": "mV", "fluxdep_fit": legacy_fit})
    dry = migrate(replace(request_data, dry_run=True), analysis_mapping)
    predictor_path = (
        request_data.results_root / request_data.name / "predictors/Qlegacy.json"
    )
    assert any(
        item.old_file == source
        and item.old_key == "fluxdep_fit.params.EJ"
        and item.new_file == predictor_path
        and item.new_path == "params.EJ"
        and item.action == "value"
        for item in dry.key_mappings
    )
    assert not (request_data.results_root / request_data.name).exists()
    assert not backup_path(request_data).exists()
    actual = migrate(request_data, analysis_mapping)
    assert actual.key_mappings == dry.key_mappings
    assert actual.pending == dry.pending
    edited = read_predictor(request_data)
    edited["flux_unit"] = "V"
    predictor_path.write_text(json.dumps(edited), encoding="utf-8")
    before = predictor_path.read_bytes()
    migrate(replace(request_data, resume=True), analysis_mapping)
    assert predictor_path.read_bytes() == before
