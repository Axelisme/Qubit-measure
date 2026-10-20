"""Legacy analysis handoff conversion through migrate_storage and ResultEntry.

Remove with the offline migrations package once legacy entries are converted.
"""

import importlib
import json
import os
import shutil
from collections.abc import Generator
from dataclasses import replace
from pathlib import Path
from typing import Literal

import pytest
from pydantic import ConfigDict, Field, JsonValue, TypeAdapter
from zcu_tools.datafile import JsonObject
from zcu_tools.resources.dispersive_file import load_dispersive_file
from zcu_tools.resources.document_store import YamlMap
from zcu_tools.resources.entry import (
    ComponentSchema,
    ResultEntry,
    component_registry,
    document_value,
    snapshot,
)
from zcu_tools.resources.predictor_file import (
    PredictorFile,
    PredictorSource,
    load_predictor_file,
)
from zcu_tools.resources.t1_curve_file import load_t1_curve_file
from zcu_tools.simulate.fluxonium.predict import FluxoniumParams

from migrations.storage_mapping import build_mapping
from migrations.storage_migration import (
    KeyRule,
    MigrationInputError,
    MigrationMapping,
    MigrationReport,
    MigrationRequest,
    migrate_storage,
    report_json,
)
from migrations.storage_migration.state import read_manifest
from tests.migrations.storage_migration.fakes import (
    manifest_path,
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
    wiring: dict[str, int] = Field(default_factory=dict)
    module: dict[str, str] = Field(default_factory=dict)
    flux_unit: Literal["A", "V"] | None = None
    flux_half: float | None = None
    flux_period: float | None = None
    flux_int: float | None = None
    flux: float | None = None


@pytest.fixture(scope="module", autouse=True)
def analysis_kind() -> Generator[None]:
    before = registry_state()
    component_registry.register("qubit/migration_analysis", AnalysisQubit)
    component_registry.register("amplifier/migration_analysis", AnalysisQubit)
    component_registry.register("qubit/migration_no_ext", ComponentSchema)
    try:
        yield
    finally:
        component_registry.unregister("qubit/migration_analysis")
        component_registry.unregister("amplifier/migration_analysis")
        component_registry.unregister("qubit/migration_no_ext")
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


def load_migrated_predictor(request: MigrationRequest) -> PredictorFile:
    """Read the migration output through the predictor format owner."""
    return load_predictor_file(
        request.results_root / request.name / "predictors" / "Qlegacy.json"
    )


def backup_path(request: MigrationRequest) -> Path:
    """Locate the explicit synthetic backup; never use production defaults."""
    if request.backup_parent is None:
        raise ValueError("Tests must supply a synthetic backup_parent")
    return request.backup_parent / "repo-backup"


def predictor_params(request: MigrationRequest) -> JsonObject:
    """Return the public predictor parameters as a typed JSON object."""
    return TypeAdapter(JsonObject).validate_python(read_predictor(request)["params"])


@pytest.mark.parametrize("dry_run", [True, False])
@pytest.mark.parametrize("unit", [None, "mA"])
def test_historical_top_level_fit_projects_exact_source_leaves(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
    dry_run: bool,
    unit: str | None,
) -> None:
    request = replace(request_data, dry_run=dry_run, assume_flux_unit=unit)
    values: JsonObject = {
        "params": {"EJ": 6.27, "EC": 1.41, "EL": 0.8, "custom.energy": [3, None]},
        "half flux": 0.25,
        "period": 1.5,
        "name": "historical author",
        "project": {"name": "old project"},
        "schema_version": "legacy",
        "timestamp": "unproven local time",
    }
    source = write_params(request, values)
    before = {p: p.read_bytes() for p in request.result_root.parent.rglob("*") if p.is_file()}
    report = migrate(request, analysis_mapping)
    converted = {
        row.source_keys: row.new_path
        for row in report.key_mappings
        if row.old_file == source and row.action == "value"
    }
    assert converted.get(("params", "EJ")) == "params.EJ"
    assert converted.get(("params", "EC")) == "params.EC"
    assert converted.get(("params", "EL")) == "params.EL"
    assert converted.get(("half flux",)) == "params.flux_half"
    assert converted.get(("period",)) == "params.flux_period"
    assert any(item.location == "timestamp" for item in report.pending)
    assert any(item.location == "flux_unit" for item in report.pending) == (unit is None)
    preserved = {
        row.source_keys: row.destination_keys
        for row in report.key_mappings
        if row.action == "preserved"
    }
    assert preserved[("params", "custom.energy")] == (
        "Qlegacy", "ext", "legacy_params", "params", "custom.energy"
    )
    if dry_run:
        assert {p: p.read_bytes() for p in request.result_root.parent.rglob("*") if p.is_file()} == before
        assert not backup_path(request).exists()
        return
    predictor = load_migrated_predictor(request)
    assert predictor == PredictorFile(
        model="fluxonium",
        params=FluxoniumParams(EJ=6.27, EC=1.41, EL=0.8, flux_half=0.25, flux_period=1.5, flux_bias=0.0),
        flux_unit=unit,
        source=PredictorSource(None, None),
    )
    assert document_value(snapshot(open_entry(request).setup), "Qlegacy.ext.legacy_params") == {
        "params": {"custom.energy": [3, None]},
        "name": "historical author",
        "project": {"name": "old project"},
        "schema_version": "legacy",
        "timestamp": "unproven local time",
    }
    migrate(replace(request, resume=True), analysis_mapping)
    assert load_migrated_predictor(request) == predictor


@pytest.mark.parametrize("energy_key", ["EJ", "EL"])
def test_fluxonium_identifying_parameter_allows_unknown_siblings(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
    legacy_fit: JsonObject,
    energy_key: str,
) -> None:
    legacy_fit["params"] = {energy_key: 0.8, "custom.energy": {"note": "untouched"}, "EC": True}
    source = write_params(request_data, {"fluxdep_fit": legacy_fit, "flux_unit": "mA"})
    report = migrate(request_data, analysis_mapping)
    predictor = request_data.results_root / request_data.name / "predictors/Qlegacy.json"
    assert predictor.is_file(), "an extra key must not block supported fluxonium data"
    assert load_predictor_file(predictor).params == FluxoniumParams.from_mapping({energy_key: 0.8, "flux_half": 0.2, "flux_period": 1.0, "flux_bias": 0.0})
    assert document_value(snapshot(open_entry(request_data).setup), "Qlegacy.ext.legacy_params.fluxdep_fit") == {
        "params": {"custom.energy": {"note": "untouched"}, "EC": True},
        "flux_int": -0.3,
    }
    assert any(item.location == "fluxdep_fit.params.EC" for item in report.pending)
    assert not any(item.location == "fluxdep_fit.params" for item in report.pending)
    assert any(row.old_file == source and row.action == "preserved" and row.source_keys == ("fluxdep_fit", "params", "custom.energy", "note") for row in report.key_mappings)


@pytest.mark.parametrize("allows", [{}, {"r_f": 5.2, "sample_f": 9.7, "transitions": [[0, 1]], "mirror": [[1, 2]], "author.note": "unchanged"}])
def test_historical_allows_are_owner_annotations_without_reinterpretation(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
    allows: JsonObject,
) -> None:
    source = write_params(request_data, {"allows": allows, "timestamp": "2020-07-07T16:30:00+08:00", "name": "old author"})
    report = migrate(request_data, analysis_mapping)
    entry = open_entry(request_data)
    target = entry.dispersive_path / "Qlegacy.json"
    assert target.is_file(), "historical allows must reach the dispersive owner"
    dispersive = load_dispersive_file(target)
    assert dispersive.params.plot_transitions.model_dump(exclude_unset=True) == allows
    assert dispersive.params.g is None
    assert dispersive.params.bare_rf is None
    assert dispersive.source.run_id is None
    assert dispersive.source.created_at == "2020-07-07T16:30:00+08:00"
    assert document_value(snapshot(entry.setup), "Qlegacy.ext.legacy_params") == {"name": "old author"}
    assert any(row.old_file == source and row.action == "value" and row.source_keys == ("allows",) and row.new_path == "params.plot_transitions" for row in report.key_mappings)


@pytest.mark.parametrize("equal", [True, False])
def test_newer_fit_wins_and_top_level_values_are_compared_by_source(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
    legacy_fit: JsonObject,
    equal: bool,
) -> None:
    values: JsonObject = {
        "flux_unit": "mA",
        "fluxdep_fit": legacy_fit,
        "params": {"EJ": 6.27 if equal else 8.2, "EC": 1.41, "EL": 0.8, "author": "keep"},
        "half flux": 0.2 if equal else 0.7,
        "period": 1.0,
    }
    source = write_params(request_data, values)
    report = migrate(request_data, analysis_mapping)
    assert load_migrated_predictor(request_data).params == FluxoniumParams(EJ=6.27, EC=1.41, EL=0.8, flux_half=0.2, flux_period=1.0, flux_bias=0.0)
    assert document_value(snapshot(open_entry(request_data).setup), "Qlegacy.ext.legacy_params") == {
        "fluxdep_fit": {"flux_int": -0.3},
        "params": {"author": "keep", **({} if equal else {"EJ": 8.2})},
        **({} if equal else {"half flux": 0.7}),
    }
    compared = [row for row in report.key_mappings if row.old_file == source and row.source_keys == ("params", "EJ")]
    assert len(compared) == 1
    assert compared[0].action == ("remove" if equal else "preserved")
    assert compared[0].compared_source_keys == ("fluxdep_fit", "params", "EJ")
    assert "fluxdep_fit" in compared[0].reason
    saved = read_manifest(manifest_path(request_data)).report
    assert [row for row in saved.key_mappings if row.source_keys == ("params", "EJ")] == compared


def test_newer_transition_annotations_preserve_only_different_historical_leaves(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
) -> None:
    source = write_params(request_data, {
        "allows": {"r_f": 5.2, "sample_f": 9.7, "transitions": [[0, 1]], "mirror": [[1, 2]], "nested": {"equal": [], "different": "old"}, "old.only": "preserve"},
        "fluxdep_fit": {"plot_transitions": {"r_f": 5.2, "sample_f": 8.1, "transitions": [[0, 1]], "mirror": [[2, 3]], "nested": {"equal": [], "different": "new"}}},
    })
    report = migrate(request_data, analysis_mapping)
    entry = open_entry(request_data)
    assert load_dispersive_file(entry.dispersive_path / "Qlegacy.json").params.plot_transitions.model_dump(exclude_unset=True) == {
        "r_f": 5.2, "sample_f": 8.1, "transitions": [[0, 1]], "mirror": [[2, 3]], "nested": {"equal": [], "different": "new"}
    }
    assert document_value(snapshot(entry.setup), "Qlegacy.ext.legacy_params") == {
        "allows": {"sample_f": 9.7, "mirror": [[1, 2]], "nested": {"different": "old"}, "old.only": "preserve"}
    }
    comparisons = {row.source_keys: row for row in report.key_mappings if row.old_file == source and row.compared_source_keys is not None}
    assert comparisons.get(("allows", "old.only")) is not None
    assert comparisons[("allows", "old.only")].action == "preserved"
    assert comparisons[("allows", "old.only")].compared_source_keys == ("fluxdep_fit", "plot_transitions", "old.only")
    assert comparisons[("allows", "r_f")].action == "remove"
    assert comparisons[("allows", "nested", "equal")].action == "remove"
    different = comparisons[("allows", "nested", "different")]
    assert different.action == "preserved"
    assert different.compared_source_keys == ("fluxdep_fit", "plot_transitions", "nested", "different")
    assert different.destination_keys == ("Qlegacy", "ext", "legacy_params", "allows", "nested", "different")


@pytest.fixture
def metadata_mapping(analysis_mapping: MigrationMapping) -> MigrationMapping:
    concrete = build_mapping(
        qubit_kind="qubit/fluxonium", data_schemas={}, native_tags={}
    )
    return replace(
        concrete,
        components={
            "Q1": {"kind": "qubit/migration_analysis"},
            "J1": {"kind": "amplifier/migration_analysis"},
        },
        roles={"qubit": "Q1"},
    )


def test_missing_units_drive_alias_and_module_scalars_remain_observable(
    request_data: MigrationRequest, metadata_mapping: MigrationMapping
) -> None:
    adapter = TypeAdapter(MigrationRequest)
    request = adapter.validate_python(
        {**adapter.dump_python(request_data), "assume_flux_unit": "mA"}
    )
    write_meta(request, "", {"qub_ch": 3, "pi_gain": 0.2, "flx_half": 0.5})
    write_meta(request, "bias", {"qub_ch": 4, "qub_0_1_ch": 2, "pi2_len": 0.03})
    report = migrate(request, metadata_mapping)
    setup = snapshot(open_entry(request).setup)
    assert document_value(setup, "Q1.wiring") == {"ch": 3}
    assert document_value(setup, "Q1.ext.pi_gain") == 0.2
    assert document_value(setup, "Q1.flux_half") == 0.5
    assert document_value(setup, "general.ext.flux_unit") == "mA"
    point = snapshot(open_entry(request).use_point("bias"))
    assert document_value(point, "Q1.wiring.ch") == 2
    assert document_value(point, "Q1.ext.pi2_len") == 0.03
    assert any("假設單位" in row.reason for row in report.key_mappings)


def test_unbound_pi_scalars_are_saved_in_component_extensions(
    request_data: MigrationRequest, metadata_mapping: MigrationMapping
) -> None:
    write_meta(request_data, "", {"pi_gain": 0.2, "pi_len": 0.05})
    migrate(request_data, metadata_mapping)
    setup = snapshot(open_entry(request_data).setup)
    assert document_value(setup, "Q1.ext") == {"pi_gain": 0.2, "pi_len": 0.05}


def test_assumption_supplies_missing_metadata_evidence_without_changing_values(
    request_data: MigrationRequest, metadata_mapping: MigrationMapping
) -> None:
    adapter = TypeAdapter(MigrationRequest)
    request = adapter.validate_python(
        {**adapter.dump_python(request_data), "assume_flux_unit": "mA"}
    )
    write_meta(request, "", {"flx_half": 0.5, "cur_A": 0.7})
    report = migrate(request, metadata_mapping)
    setup = snapshot(open_entry(request).setup)
    assert document_value(setup, "general.ext") == {
        "flux_unit": "mA",
        "flux_value": 0.7,
    }
    assert document_value(setup, "Q1.flux_half") == 0.5
    assert document_value(setup, "general.ext.flux_value") == 0.7
    assert any("假設單位" in row.reason for row in report.key_mappings)


@pytest.mark.parametrize("unit_source", ["meta_info.json", "params.json"])
def test_entry_explicit_unit_beats_assumption_and_reports_its_file(
    request_data: MigrationRequest,
    metadata_mapping: MigrationMapping,
    unit_source: str,
) -> None:
    adapter = TypeAdapter(MigrationRequest)
    request = adapter.validate_python(
        {**adapter.dump_python(request_data), "assume_flux_unit": "mA"}
    )
    root = request.result_root / "chip/qubit"
    (root / unit_source).write_text('{"flux_unit": "V"}', encoding="utf-8")
    write_meta(request, "bias", {"flx_half": 0.3})
    report = migrate(request, metadata_mapping)
    point = snapshot(open_entry(request).use_point("bias"))
    assert document_value(point, "general.ext.flux_unit") == "V"
    assert document_value(point, "Q1.flux_half") == 0.3
    assert any(
        row.old_key == "flx_half" and str(root / unit_source) in row.reason
        for row in report.key_mappings
    )


def test_point_unit_override_clears_only_unreplaced_inherited_group_values(
    request_data: MigrationRequest, metadata_mapping: MigrationMapping
) -> None:
    metadata_mapping = replace(
        metadata_mapping,
        rules=(
            *metadata_mapping.rules,
            KeyRule(
                old_key="period_error",
                target_path="Q1.flux_period",
                action="stderr",
                reason="Synthetic uncertainty in the value working unit",
            ),
        ),
    )
    write_meta(
        request_data,
        "",
        {
            "flux_unit": "V",
            "local_flux_unit": "A",
            "flx_half": 0.2,
            "flx_period": 0.4,
            "period_error": 0.02,
            "flx_int": 0.1,
            "cur_A": 0.5,
            "flux_bias": 0.6,
            "jpa_flux_unit": "V",
            "best_jpa_flux": 1.0,
        },
    )
    point_meta = write_meta(
        request_data,
        "bias",
        {"flux_unit": "A", "flx_half": 0.3, "local_flux_unit": "V"},
    )
    report = migrate(request_data, metadata_mapping)
    point = snapshot(open_entry(request_data).use_point("bias"))
    assert document_value(point, "Q1.flux_half") == 0.3
    assert document_value(point, "Q1.flux_period") is None
    assert open_entry(request_data).setup.meta("Q1.flux_period") == {"stderr": 0.02}
    assert open_entry(request_data).use_point("bias").meta("Q1.flux_period") == {}
    assert document_value(point, "Q1.flux_int") is None
    assert document_value(point, "general.ext") == {"flux_unit": "A"}
    assert document_value(point, "Q1.ext") == {}
    assert document_value(point, "Q1.flux_unit") == "V"
    assert document_value(point, "J1.flux") == 1.0
    assert document_value(point, "J1.flux_unit") == "V"
    assert not any(item.source == point_meta for item in report.pending)


def migrate_with_unit_pending(
    request: MigrationRequest, mapping: MigrationMapping
) -> MigrationReport:
    """Unit incompatibility is located pending, not an entry-wide schema abort."""
    try:
        return migrate(request, mapping)
    except ValueError as exc:
        pytest.fail(f"Unit incompatibility must become located pending: {exc}")


def test_jpa_does_not_borrow_params_unit_and_rejects_unsupported_owner_unit(
    request_data: MigrationRequest, metadata_mapping: MigrationMapping
) -> None:
    write_params(request_data, {"flux_unit": "A"})
    source = write_meta(
        request_data,
        "",
        {
            "best_jpa_flux": 0.2,
            "local_flux_unit": "mA",
            "flx_half": 0.3,
        },
    )
    report = migrate_with_unit_pending(request_data, metadata_mapping)
    setup = snapshot(open_entry(request_data).setup)
    assert any(
        item.source == source and item.location == "best_jpa_flux"
        for item in report.pending
    )
    assert any(
        item.source == source and item.location == "local_flux_unit"
        for item in report.pending
    )
    assert document_value(setup, "Q1.flux_half") == 0.3
    assert getattr(open_entry(request_data).setup.J1, "flux", None) is None


def test_invalid_jpa_unit_keeps_whole_group_pending_without_conversion(
    request_data: MigrationRequest, metadata_mapping: MigrationMapping
) -> None:
    source = write_meta(
        request_data,
        "",
        {
            "jpa_flux_unit": "mA",
            "best_jpa_flux": 2.0,
        },
    )
    report = migrate_with_unit_pending(request_data, metadata_mapping)
    assert {item.location for item in report.pending if item.source == source} >= {
        "jpa_flux_unit",
        "best_jpa_flux",
    }
    assert getattr(open_entry(request_data).setup.J1, "flux", None) is None


def test_conflicting_entry_units_block_values_instead_of_using_assumption(
    request_data: MigrationRequest, metadata_mapping: MigrationMapping
) -> None:
    adapter = TypeAdapter(MigrationRequest)
    request = adapter.validate_python(
        {**adapter.dump_python(request_data), "assume_flux_unit": "mA"}
    )
    write_params(request, {"flux_unit": "A"})
    source = write_meta(request, "", {"flux_unit": "V", "flx_half": 0.2})
    report = migrate(request, metadata_mapping)
    setup = snapshot(open_entry(request).setup)
    assert document_value(setup, "general.ext") == {}
    assert any(
        item.source == source
        and item.location == "flx_half"
        and "params.json" in item.reason
        for item in report.pending
    )


def declare_request(
    request: MigrationRequest, **declarations: JsonValue
) -> MigrationRequest:
    """Apply operator input through the request's typed boundary."""
    adapter = TypeAdapter(MigrationRequest)
    return adapter.validate_python({**adapter.dump_python(request), **declarations})


def test_jpa_never_borrows_global_assumption(
    request_data: MigrationRequest, metadata_mapping: MigrationMapping
) -> None:
    request = declare_request(request_data, assume_flux_unit="A")
    source = write_meta(request, "", {"cur_A": 0.4, "best_jpa_flux": 0.2})
    report = migrate(request, metadata_mapping)
    assert any(
        item.source == source and item.location == "best_jpa_flux"
        for item in report.pending
    )
    assert getattr(open_entry(request).setup.J1, "flux", None) is None
    assert document_value(snapshot(open_entry(request).setup), "general.ext") == {
        "flux_unit": "A",
        "flux_value": 0.4,
    }


def test_independent_jpa_assumption_publishes_v_without_changing_global_ma(
    request_data: MigrationRequest, metadata_mapping: MigrationMapping
) -> None:
    request = declare_request(
        request_data, assume_flux_unit="mA", assume_jpa_flux_unit="V"
    )
    write_meta(request, "", {"cur_A": 0.4, "best_jpa_flux": 0.2})
    report = migrate(request, metadata_mapping)
    assert getattr(open_entry(request).setup.J1, "flux", None) == 0.2
    setup = snapshot(open_entry(request).setup)
    assert document_value(setup, "J1.flux_unit") == "V"
    assert document_value(setup, "general.ext") == {
        "flux_unit": "mA",
        "flux_value": 0.4,
    }
    assert any(
        row.old_key == "jpa_flux_unit" and "假設單位" in row.reason
        for row in report.key_mappings
    )


@pytest.mark.parametrize("explicit", ["A", "mA", None])
def test_jpa_assumption_never_replaces_explicit_or_invalid_evidence(
    request_data: MigrationRequest,
    metadata_mapping: MigrationMapping,
    explicit: str | None,
) -> None:
    request = declare_request(request_data, assume_jpa_flux_unit="V")
    source = write_meta(request, "", {"jpa_flux_unit": explicit, "best_jpa_flux": 0.2})
    report = migrate_with_unit_pending(request, metadata_mapping)
    if explicit == "A":
        assert (
            document_value(snapshot(open_entry(request).setup), "J1.flux_unit") == "A"
        )
        assert not any(item.source == source for item in report.pending)
    else:
        assert getattr(open_entry(request).setup.J1, "flux", None) is None
        assert any(
            item.source == source and item.location == "best_jpa_flux"
            for item in report.pending
        )


def test_resume_rejects_changed_jpa_assumption_before_writes(
    request_data: MigrationRequest, metadata_mapping: MigrationMapping
) -> None:
    request = declare_request(request_data, assume_jpa_flux_unit="V")
    write_meta(request, "", {"best_jpa_flux": 0.2})
    migrate(request, metadata_mapping)
    before = (request.results_root / request.name / "setup.yaml").read_bytes()
    changed = declare_request(request, assume_jpa_flux_unit="A", resume=True)
    with pytest.raises(MigrationInputError, match="declarations"):
        migrate(changed, metadata_mapping)
    assert (request.results_root / request.name / "setup.yaml").read_bytes() == before


@pytest.mark.parametrize("unit", ["mA", "unknown", "", True])
def test_invalid_jpa_assumption_is_rejected_before_any_entry_or_backup(
    request_data: MigrationRequest, metadata_mapping: MigrationMapping, unit: JsonValue
) -> None:
    with pytest.raises(ValueError, match="assume_jpa_flux_unit"):
        migrate(
            declare_request(request_data, assume_jpa_flux_unit=unit), metadata_mapping
        )
    assert not (request_data.results_root / request_data.name).exists()
    assert not backup_path(request_data).exists()


def test_sample_declarations_publish_runtime_rows_and_resume_once(
    request_data: MigrationRequest, analysis_mapping: MigrationMapping
) -> None:
    adapter = TypeAdapter(MigrationRequest)
    request = adapter.validate_python(
        {
            **adapter.dump_python(request_data),
            "sample_value_column": "calibrated mA",
            "sample_source_unit": "mA",
        }
    )
    source = request.result_root / "chip/qubit/samples.csv"
    original = b"calibrated mA,T1 (us)\r\n2,50\r\nbad,60\r\n"
    source.write_bytes(original)
    dry = migrate(replace(request, dry_run=True), analysis_mapping)
    assert not (request.results_root / request.name).exists()
    report = migrate(request, analysis_mapping)
    table = open_entry(request).samples("Qlegacy")
    assert len(table.read()) == 1
    row = table.latest("legacy-samples-1")
    assert row is not None
    assert row["dev_value"] == 0.002
    assert row["dev_unit"] == "A"
    assert row["t1_us"] == 50
    assert any(item.location == "row 2" for item in report.pending)
    raw_csv = open_entry(request).result_path / "samples.csv"
    assert raw_csv.is_file(), "partial conversion must also preserve the original CSV"
    assert raw_csv.read_bytes() == original
    preserved = next(item for item in report.preserved_files if item.source == source)
    assert preserved.destination == raw_csv
    assert preserved.source_hash == preserved.destination_hash
    assert dry.pending == report.pending
    before = table.path.read_bytes()
    migrate(replace(request, resume=True), analysis_mapping)
    assert table.path.read_bytes() == before
    assert raw_csv.read_bytes() == original


def test_unknown_csv_fields_survive_storage_publication_reports_and_completed_edits(
    request_data: MigrationRequest, analysis_mapping: MigrationMapping
) -> None:
    request = replace(
        request_data, sample_value_column="calibrated mA", sample_source_unit="mA"
    )
    source = request.result_root / "chip/qubit/samples.csv"
    original = (
        b"calibrated mA,T1 (us),comment,custom,empty\r\n"
        b"2,50,Manual Added, 001 ,\r\n3,60,Manual Added,NaN,\r\n"
    )
    source.write_bytes(original)
    before = {
        path: path.read_bytes()
        for root in (request.result_root, request.database_root)
        for path in root.rglob("*")
        if path.is_file()
    }
    dry = migrate_storage(
        replace(request, dry_run=True),
        mapping=analysis_mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    assert not any(item.source == source for item in dry.pending)
    assert len(dry.preserved_sample_fields) == 6
    assert not (request.results_root / request.name).exists()
    assert dry.backup_path is not None and not dry.backup_path.exists()
    assert {path: path.read_bytes() for path in before} == before
    report = migrate_storage(
        request,
        mapping=analysis_mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    assert report.preserved_sample_fields == dry.preserved_sample_fields
    table = open_entry(request).samples("Qlegacy")
    assert list(table.read()["comment"]) == ["Manual Added", "Manual Added"]
    assert list(table.read()["custom"]) == [" 001 ", "NaN"]
    assert list(table.read()["empty"]) == ["", ""]
    assert list(table.read()["dev_value"]) == [0.002, 0.003]
    assert list(table.read()["t1_us"]) == [50, 60]
    assert [
        (
            item.field.source_path,
            item.field.row_number,
            item.field.column,
            item.field.value,
            item.field.run_id,
            item.output_path,
        )
        for item in report.preserved_sample_fields
    ] == [
        (source, number, column, value, f"legacy-samples-{number}", table.path)
        for number, custom in [(1, " 001 "), (2, "NaN")]
        for column, value in [
            ("comment", "Manual Added"),
            ("custom", custom),
            ("empty", ""),
        ]
    ]
    assert not any(item.source == source for item in report.pending)
    assert report.backup_path is not None
    assert (
        report.backup_path / "result/chip/qubit/samples.csv"
    ).read_bytes() == original
    assert not source.parent.exists()
    assert not (request.database_root / "chip/qubit").exists()
    published = table.path.read_bytes()
    resumed = migrate_storage(
        replace(request, resume=True),
        mapping=analysis_mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    assert resumed.preserved_sample_fields == report.preserved_sample_fields
    assert table.path.read_bytes() == published
    table.append(run_id="legacy-samples-1", comment="user edit")
    edited = table.path.read_bytes()
    with pytest.raises(MigrationInputError, match="hash changed"):
        migrate_storage(
            replace(request, resume=True),
            mapping=analysis_mapping,
            validate_cfg=validate_cfg,
            validate_native=noop_validation,
        )
    assert table.path.read_bytes() == edited
    persisted = json.loads(manifest_path(request).read_text())["report"]
    assert persisted["preserved_sample_fields"][0]["field"]["value"] == "Manual Added"


def test_sample_prepared_publication_resumes_without_duplicate_rows(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = TypeAdapter(MigrationRequest)
    request = adapter.validate_python(
        {
            **adapter.dump_python(request_data),
            "sample_value_column": "calibrated mA",
            "sample_source_unit": "mA",
        }
    )
    source = request.result_root / "chip/qubit/samples.csv"
    source.write_text("calibrated mA,T1 (us)\n2,50\n", encoding="utf-8")
    output = request.results_root / request.name / "samples/Qlegacy.jsonl"
    link = os.link

    def fail_sample_publish(src: Path, dst: Path) -> None:
        if dst == output:
            raise OSError("sample publish interrupted")
        link(src, dst)

    with monkeypatch.context() as patch:
        patch.setattr(os, "link", fail_sample_publish)
        with pytest.raises(OSError, match="sample publish interrupted"):
            migrate(request, analysis_mapping)
    migrate(replace(request, resume=True), analysis_mapping)
    table = open_entry(request).samples("Qlegacy")
    assert len(table.read()) == 1
    assert len(output.read_text().splitlines()) == 1


def test_resume_refuses_changed_sample_declarations_before_writes(
    request_data: MigrationRequest, analysis_mapping: MigrationMapping
) -> None:
    adapter = TypeAdapter(MigrationRequest)
    request = adapter.validate_python(
        {
            **adapter.dump_python(request_data),
            "sample_value_column": "calibrated mA",
            "sample_source_unit": "mA",
        }
    )
    source = request.result_root / "chip/qubit/samples.csv"
    source.write_text("calibrated mA,T1 (us)\n2,50\n", encoding="utf-8")
    migrate(request, analysis_mapping)
    changed = adapter.validate_python(
        {
            **adapter.dump_python(request),
            "sample_source_unit": "A",
            "resume": True,
        }
    )
    with pytest.raises(MigrationInputError, match="declarations"):
        migrate(changed, analysis_mapping)


def test_sample_missing_declarations_are_located_without_using_flux_assumption(
    request_data: MigrationRequest, analysis_mapping: MigrationMapping
) -> None:
    request = declare_request(
        request_data, assume_flux_unit="mA", assume_jpa_flux_unit="V"
    )
    source = request.result_root / "chip/qubit/samples.csv"
    source.write_text("calibrated mA,T1 (us)\n2,50\n", encoding="utf-8")
    report = migrate(request, analysis_mapping)
    assert any(
        item.source == source and item.location == "row 1" for item in report.pending
    )
    assert open_entry(request_data).samples("Qlegacy").read().empty


def test_assumed_unit_fills_only_missing_predictor_evidence(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
    legacy_fit: JsonObject,
) -> None:
    adapter = TypeAdapter(MigrationRequest)
    request = adapter.validate_python(
        {**adapter.dump_python(request_data), "assume_flux_unit": "mA"}
    )
    write_params(request, {"fluxdep_fit": legacy_fit})
    report = migrate(request, analysis_mapping)
    assert read_predictor(request)["flux_unit"] == "mA"
    assert any(
        row.new_path == "flux_unit" and "假設單位" in row.reason
        for row in report.key_mappings
    )
    assert not any(item.location == "fluxdep_fit.flux_unit" for item in report.pending)


@pytest.mark.parametrize(
    "values",
    [
        {"flux_unit": "V"},
        {"flux_unit": "A", "fluxdep_fit": {"flux_unit": "V"}},
        {"flux_unit": None},
        {"flux_unit": "unsupported"},
    ],
)
def test_assumption_never_replaces_explicit_predictor_units(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
    legacy_fit: JsonObject,
    values: JsonObject,
) -> None:
    adapter = TypeAdapter(MigrationRequest)
    request = adapter.validate_python(
        {**adapter.dump_python(request_data), "assume_flux_unit": "mA"}
    )
    fit = values.pop("fluxdep_fit", {})
    legacy_fit.update(TypeAdapter(JsonObject).validate_python(fit))
    values["fluxdep_fit"] = legacy_fit
    write_params(request, values)
    migrate(request, analysis_mapping)
    assert read_predictor(request)["flux_unit"] == (
        "V" if values.get("flux_unit") == "V" else None
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


def test_analysis_files_publish_typed_unknowns_and_keep_original_sections(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
    analysis_sections: JsonObject,
) -> None:
    write_params(request_data, analysis_sections)
    report = migrate(request_data, analysis_mapping)
    entry = open_entry(request_data)
    for kind in ("dispersive", "t1_curve"):
        file = entry.result_path / kind / "Qlegacy.json"
        assert file.is_file(), f"migration must publish {kind} fit"
        owner = importlib.import_module(f"zcu_tools.resources.{kind}_file")
        loaded = getattr(owner, f"load_{kind}_file")(file)
        assert loaded.source.run_id is None
        assert any(
            row.new_file == file and row.action == "value"
            for row in report.key_mappings
        )
        if kind == "dispersive":
            assert entry.dispersive_path == file.parent
            assert loaded.params.g is None
            assert loaded.params.bare_rf == 5.1
            assert loaded.params.plot_transitions.r_f == 5.2
            assert loaded.params.plot_transitions.model_extra == {"0.1": [1, 2]}
            assert loaded.source.created_at is None
            assert any(
                item.location == "dispersive.timestamp" for item in report.pending
            )
        else:
            assert entry.t1_curve_path == file.parent
            assert loaded.params.params.Q_cap is None
            assert loaded.params.params.Temp == 0.04
            assert loaded.params.stderr.Q_cap is None
            assert loaded.params.bounds == {"Q_cap": (1.0, None)}
            assert loaded.source.created_at == "2020-07-07T16:30:00+08:00"
    doc = snapshot(entry.setup)
    assert document_value(doc, "Qlegacy.ext.legacy_params") == {
        "dispersive": {"timestamp": "unproven", "future": [1, None]}
    }


@pytest.mark.parametrize(
    ("section", "kind", "content"),
    [
        ("dispersive", "dispersive", {"g": None, "timestamp": "unproven"}),
        ("t1_curve_fit", "t1_curve", {"params": {"Q_cap": None}, "timestamp": None}),
        (
            "fluxdep_fit",
            "dispersive",
            {"plot_transitions": {"r_f": 5.2}, "timestamp": "2020-07-07T16:30:00"},
        ),
    ],
)
def test_analysis_timestamp_locators_survive_source_cleanup_and_resume(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
    legacy_fit: JsonObject,
    section: str,
    kind: str,
    content: JsonObject,
) -> None:
    legacy_fit["timestamp"] = "unproven"
    values: JsonObject = {"flux_unit": "A", "fluxdep_fit": legacy_fit}
    if section == "fluxdep_fit":
        legacy_fit.update(content)
    else:
        values[section] = content
    source = write_params(request_data, values)
    original = source.read_bytes()
    destination = request_data.results_root / request_data.name

    dry = migrate_storage(
        replace(request_data, dry_run=True),
        mapping=analysis_mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    assert source.read_bytes() == original
    assert not destination.exists()
    assert not backup_path(request_data).exists()

    report = migrate_storage(
        request_data,
        mapping=analysis_mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    backup_source = backup_path(request_data) / "result/chip/qubit/params.json"
    assert backup_source.read_bytes() == original
    assert TypeAdapter(JsonObject).validate_json(backup_source.read_text()) == values
    assert not source.parent.exists()
    assert not (request_data.database_root / "chip/qubit").exists()
    file = destination / kind / "Qlegacy.json"
    loaded = (
        load_dispersive_file(file) if kind == "dispersive" else load_t1_curve_file(file)
    )
    assert loaded.source.created_at is None
    assert load_migrated_predictor(request_data).source.created_at is None
    locations = {f"{section}.timestamp", "fluxdep_fit.timestamp"}
    rows = [
        row
        for row in report.key_mappings
        if row.old_key in locations and row.action == "pending"
    ]
    assert any(
        row.new_file == file and row.old_key == f"{section}.timestamp" for row in rows
    )
    for row in rows:
        assert row.old_file == source
        assert f"{backup_source}#{row.old_key}" in row.reason
    pending = [item for item in report.pending if item.location in locations]
    assert pending
    for item in pending:
        assert item.source == backup_source
        assert f"{backup_source}#{item.location}" in item.reason
    dry_pending = [item for item in dry.pending if item.location in locations]
    assert [(item.location, item.reason) for item in dry_pending] == [
        (item.location, item.reason) for item in pending
    ]
    assert all(item.source == source for item in dry_pending)

    published = file.read_bytes()
    resumed = migrate_storage(
        replace(request_data, resume=True),
        mapping=analysis_mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    assert resumed.backup_path == report.backup_path == backup_path(request_data)
    assert resumed.pending == report.pending
    assert file.read_bytes() == published
    assert backup_source.read_bytes() == original


def test_transition_only_fit_publishes_dispersive_without_inventing_fit_values(
    request_data: MigrationRequest, analysis_mapping: MigrationMapping
) -> None:
    write_params(request_data, {"fluxdep_fit": {"plot_transitions": {"r_f": 5.2}}})
    migrate(request_data, analysis_mapping)
    file = open_entry(request_data).result_path / "dispersive/Qlegacy.json"
    assert file.is_file(), "transition seed alone must publish dispersive data"
    owner = importlib.import_module("zcu_tools.resources.dispersive_file")
    loaded = owner.load_dispersive_file(file)
    assert loaded.params.g is None
    assert loaded.params.bare_rf is None
    assert loaded.params.plot_transitions.r_f == 5.2


@pytest.mark.parametrize(
    ("section", "value", "kind"),
    [
        ("dispersive", {"g": True}, "dispersive"),
        ("t1_curve_fit", {"params": {"Temp": None}, "free": ["Q_cap"]}, "t1_curve"),
    ],
)
def test_invalid_analysis_content_stays_located_pending(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
    section: str,
    value: JsonObject,
    kind: str,
) -> None:
    write_params(request_data, {section: value})
    report = migrate(request_data, analysis_mapping)
    file = open_entry(request_data).result_path / kind / "Qlegacy.json"
    assert any(
        row.old_key == section
        and row.new_file == file
        and row.action == "pending"
        and "validation" in row.reason.lower()
        for row in report.key_mappings
    )
    assert (
        document_value(
            snapshot(open_entry(request_data).setup),
            f"Qlegacy.ext.legacy_params.{section}",
        )
        == value
    )


@pytest.mark.parametrize("kind", ["dispersive", "t1_curve"])
@pytest.mark.parametrize("conflict", [False, True])
def test_analysis_file_resume_validates_unfinished_and_preserves_completed_edits(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
    analysis_sections: JsonObject,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    conflict: bool,
) -> None:
    write_params(request_data, analysis_sections)
    write_meta(request_data, "bias", {})
    destination = request_data.results_root / request_data.name
    copyfile = shutil.copyfile

    def interrupt(src: Path, dst: Path, *, follow_symlinks: bool = True) -> Path:
        if src == destination / "setup.yaml" and dst.name == "point.yaml":
            raise OSError("analysis checkpoint interrupted")
        return copyfile(src, dst, follow_symlinks=follow_symlinks)

    with monkeypatch.context() as patch:
        patch.setattr(shutil, "copyfile", interrupt)
        with pytest.raises(OSError, match="analysis checkpoint interrupted"):
            migrate(request_data, analysis_mapping)
    file = destination / kind / "Qlegacy.json"
    assert file.is_file(), "fit must be published before the interrupted checkpoint"
    raw = json.loads(file.read_text(encoding="utf-8"))
    if conflict:
        raw["version"] = True
    file.write_text(json.dumps(raw, separators=(",", ":")), encoding="utf-8")
    before = file.read_bytes()
    if conflict:
        with pytest.raises(MigrationInputError, match=kind.replace("_", "[ _-]?")):
            migrate(replace(request_data, resume=True), analysis_mapping)
    else:
        migrate(replace(request_data, resume=True), analysis_mapping)
        assert file.read_bytes() == before
        raw["source"]["run_id"] = "user-edited"
        file.write_text(json.dumps(raw), encoding="utf-8")
        edited = file.read_bytes()
        migrate(replace(request_data, resume=True), analysis_mapping)
        assert file.read_bytes() == edited


def test_unfinished_t1_resume_rejects_new_active_null_channel(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
    analysis_sections: JsonObject,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_params(request_data, analysis_sections)
    write_meta(request_data, "bias", {})
    destination = request_data.results_root / request_data.name
    copyfile = shutil.copyfile

    def interrupt(src: Path, dst: Path, *, follow_symlinks: bool = True) -> Path:
        if src == destination / "setup.yaml" and dst.name == "point.yaml":
            raise OSError("T1 checkpoint interrupted")
        return copyfile(src, dst, follow_symlinks=follow_symlinks)

    with monkeypatch.context() as patch:
        patch.setattr(shutil, "copyfile", interrupt)
        with pytest.raises(OSError, match="T1 checkpoint interrupted"):
            migrate(request_data, analysis_mapping)
    file = open_entry(request_data).result_path / "t1_curve/Qlegacy.json"
    raw = json.loads(file.read_text(encoding="utf-8"))
    raw["params"]["params"]["Q_ind"] = None
    file.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(MigrationInputError, match="differs"):
        migrate(replace(request_data, resume=True), analysis_mapping)


def test_analysis_dry_run_reports_the_same_files_without_writes(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
    analysis_sections: JsonObject,
) -> None:
    write_params(request_data, analysis_sections)
    dry = migrate(replace(request_data, dry_run=True), analysis_mapping)
    destination = request_data.results_root / request_data.name
    for kind in ("dispersive", "t1_curve"):
        assert any(
            row.new_file == destination / kind / "Qlegacy.json"
            and row.action == "value"
            for row in dry.key_mappings
        )
    assert not destination.exists()
    assert not backup_path(request_data).exists()


def test_raw_params_preservation_is_typed_located_and_durable(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
    legacy_fit: JsonObject,
    analysis_sections: JsonObject,
) -> None:
    write_meta(request_data, "bias", {})
    # Keep real evidence defects alongside successful analysis publications.
    legacy_fit["timestamp"] = "unproven"
    legacy_fit["params"] = {"EJ": True, "EC": 1.41, "EL": 0.8}
    values: JsonObject = {
        **analysis_sections,
        "fluxdep_fit": legacy_fit,
        "unknown.section": {"list": [None, False, "raw"], "x.y": 7},
        "unknown": {"section": "a different literal key"},
        "project": {"chip": "original", "timestamp": None},
    }
    source = write_params(request_data, values)
    dry = migrate(replace(request_data, dry_run=True), analysis_mapping)
    preserved = [row for row in dry.key_mappings if row.action == "preserved"]
    assert {row.source_keys for row in preserved} == {
        ("dispersive", "timestamp"),
        ("dispersive", "future"),
        ("fluxdep_fit", "params", "EJ"),
        ("fluxdep_fit", "flux_int"),
        ("fluxdep_fit", "timestamp"),
        ("unknown.section", "list"),
        ("unknown.section", "x.y"),
        ("unknown", "section"),
        ("project", "chip"),
        ("project", "timestamp"),
    }, "Every remaining raw leaf needs a typed literal preservation decision"
    for row in preserved:
        assert row.old_file == source
        assert (
            row.new_file == request_data.results_root / request_data.name / "setup.yaml"
        )
        assert row.new_path is None
        assert row.source_keys is not None
        assert row.destination_keys == (
            "Qlegacy",
            "ext",
            "legacy_params",
            *row.source_keys,
        )
    expected_pending = {
        "fluxdep_fit.params.EJ",
        "fluxdep_fit.flux_unit",
        "fluxdep_fit.timestamp",
        "dispersive.timestamp",
    }
    assert {
        item.location for item in dry.pending if item.source == source
    } == expected_pending
    assert not (request_data.results_root / request_data.name).exists()
    assert not backup_path(request_data).exists()

    unavailable = migrate(
        replace(request_data, dry_run=True),
        replace(
            analysis_mapping,
            components={"Qlegacy": {"kind": "qubit/migration_no_ext"}},
        ),
    )
    assert not any(row.action == "preserved" for row in unavailable.key_mappings)
    assert {item.location for item in unavailable.pending if item.source == source} >= {
        "unknown.section.list",
        "unknown.section.x.y",
        "unknown.section",
        "project.chip",
    }

    report = migrate(request_data, analysis_mapping)
    assert report.key_mappings == dry.key_mappings
    assert report.pending == dry.pending
    entry = open_entry(request_data)
    for view in (entry.setup, entry.use_point("bias")):
        assert document_value(snapshot(view), "Qlegacy.ext.legacy_params") == {
            "dispersive": {"timestamp": "unproven", "future": [1, None]},
            "fluxdep_fit": {
                "params": {"EJ": True},
                "flux_int": -0.3,
                "timestamp": "unproven",
            },
            "unknown.section": values["unknown.section"],
            "unknown": values["unknown"],
            "project": values["project"],
        }
    predictor = load_migrated_predictor(request_data)
    assert predictor.params == FluxoniumParams(
        EC=1.41, EL=0.8, flux_half=0.2, flux_period=1.0, flux_bias=0.0
    )
    assert predictor.flux_unit is None
    assert predictor.source.created_at is None
    assert (
        load_dispersive_file(entry.dispersive_path / "Qlegacy.json").params.bare_rf
        == 5.1
    )
    assert load_t1_curve_file(entry.t1_curve_path / "Qlegacy.json").params.model_dump(
        exclude_unset=True
    )["params"] == {"Q_cap": None, "Temp": 0.04}

    # Both report encoding and durable manifest decoding must retain typed status/keys.
    encoded = report_json(report)
    restored = TypeAdapter(MigrationReport).validate_json(
        json.dumps(encoded), strict=True
    )
    assert restored.key_mappings == report.key_mappings
    manifest = read_manifest(manifest_path(request_data))
    assert manifest.report.key_mappings == report.key_mappings
    for view in (entry.setup, entry.use_point("bias")):
        with view.edit() as draft:
            draft.set("Qlegacy.ext.legacy_params", {"user.edit": ["keep"]})
    paths = [
        entry.result_path / "setup.yaml",
        entry.result_path / "points/bias/point.yaml",
    ]
    before = [path.read_bytes() for path in paths]
    resumed = migrate(replace(request_data, resume=True), analysis_mapping)
    assert resumed.key_mappings == report.key_mappings
    assert resumed.pending == report.pending
    assert [path.read_bytes() for path in paths] == before


def test_formal_analysis_values_leave_only_unconverted_raw_leaves(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
) -> None:
    timestamp = "2020-07-07T16:30:00+08:00"
    write_meta(request_data, "bias", {})
    write_params(
        request_data,
        {
            "flux_unit": "A",
            "fluxdep_fit": {
                "params": {"EJ": 6.0, "EC": None, "EL": 0.8},
                "flux_half": 0.2,
                "flux_period": 1.0,
                "timestamp": timestamp,
                "plot_transitions": {"r_f": 5.2, "literal.key": [1, None]},
            },
            "dispersive": {
                "g": 0.03,
                "bare_rf": 5.1,
                "timestamp": timestamp,
                "future.key": {"g": 0.03, "empty": {}, "list": []},
            },
            "t1_curve_fit": {
                "params": {"Temp": 0.04},
                "fixed": ["Temp"],
                "timestamp": timestamp,
            },
            "metadata": {"EJ": 6.0},
        },
    )
    report = migrate(request_data, analysis_mapping)
    entry = open_entry(request_data)
    expected = {
        "dispersive": {"future.key": {"g": 0.03, "empty": {}, "list": []}},
        "metadata": {"EJ": 6.0},
    }
    for view in (entry.setup, entry.use_point("bias")):
        assert (
            document_value(snapshot(view), "Qlegacy.ext.legacy_params") == expected
        ), (
            "Converted source leaves must live only in formal analysis files; keep unknown leaves"
        )
    assert load_migrated_predictor(request_data).params == FluxoniumParams(
        EJ=6.0, EL=0.8, flux_half=0.2, flux_period=1.0, flux_bias=0.0
    )
    dispersive = load_dispersive_file(entry.dispersive_path / "Qlegacy.json")
    assert dispersive.params.g == 0.03
    assert dispersive.params.plot_transitions is not None
    assert dispersive.params.plot_transitions.model_extra == {"literal.key": [1, None]}
    t1 = load_t1_curve_file(entry.t1_curve_path / "Qlegacy.json")
    assert t1.source.created_at == timestamp
    assert not report.pending


@pytest.mark.parametrize(
    ("values", "formal_kind", "source_keys"),
    [
        ({"dispersive": {}}, "dispersive", ("dispersive",)),
        ({"t1_curve_fit": {}}, "t1_curve", ("t1_curve_fit",)),
        (
            {
                "fluxdep_fit": {
                    "params": {},
                    "flux_unit": "A",
                    "timestamp": "2020-07-07T16:30:00+08:00",
                }
            },
            "predictors",
            ("fluxdep_fit", "params"),
        ),
    ],
)
def test_successful_empty_analysis_containers_are_not_preserved_twice(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
    values: JsonObject,
    formal_kind: str,
    source_keys: tuple[str, ...],
) -> None:
    values.update({"unknown.empty": {}, "unknown_list": []})
    source = write_params(request_data, values)
    report = migrate(request_data, analysis_mapping)
    entry = open_entry(request_data)
    assert document_value(snapshot(entry.setup), "Qlegacy.ext.legacy_params") == {
        "unknown.empty": {},
        "unknown_list": [],
    }, "Only successfully transferred known empty containers should leave raw storage"
    formal = entry.result_path / formal_kind / "Qlegacy.json"
    assert any(
        row.action == "value"
        and row.source_keys == source_keys
        and row.old_file == source
        and row.new_file == formal
        and row.new_path == "params"
        for row in report.key_mappings
    )
    if formal_kind == "predictors":
        assert load_predictor_file(formal).params == FluxoniumParams(flux_bias=0.0)
    elif formal_kind == "dispersive":
        assert load_dispersive_file(formal).params.model_dump(exclude_unset=True) == {}
    else:
        assert load_t1_curve_file(formal).params.model_dump(exclude_unset=True) == {}


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
        assert document_value(doc, "Qlegacy.ext.legacy_params.dispersive") == {
            "timestamp": "original",
            "future": {"x": True},
        }
        assert document_value(doc, "Qlegacy.ext.legacy_params.project") == {
            "chip": "old"
        }
    assert any(
        item.source_keys == ("dispersive", "future", "x")
        and item.destination_keys
        == ("Qlegacy", "ext", "legacy_params", "dispersive", "future", "x")
        and item.action == "preserved"
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
    source = write_params(
        request_data, {"t1_curve_fit": {"params": {"Q_cap": 1}, "note": "keep"}}
    )
    dry = migrate(replace(request_data, dry_run=True), analysis_mapping)
    assert any(
        item.old_file == source
        and item.source_keys == ("t1_curve_fit", "note")
        and item.destination_keys
        == ("Qlegacy", "ext", "legacy_params", "t1_curve_fit", "note")
        and item.action == "preserved"
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
    assert load_migrated_predictor(request_data) == PredictorFile(
        model="fluxonium",
        params=FluxoniumParams(
            EJ=6.27,
            EC=1.41,
            EL=0.8,
            flux_half=0.2,
            flux_period=1.0,
            flux_bias=0.0,
        ),
        flux_unit=root_unit or fit_unit,
        source=PredictorSource(None, "2020-07-07T16:30:00+08:00"),
    )
    point = snapshot(open_entry(request_data).use_point("bias"))
    assert document_value(point, "Qlegacy.EJ") == 11.0
    assert document_value(point, "general.ext.flux_unit") == "mV"
    assert document_value(point, "Qlegacy.ext.legacy_params.fluxdep_fit") == {
        "flux_int": -0.3
    }


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
    assert load_migrated_predictor(request_data).source == PredictorSource(None, None)
    backup_source = backup_path(request_data) / "result/chip/qubit/params.json"
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
    assert load_migrated_predictor(request_data) == PredictorFile(
        model="fluxonium",
        params=FluxoniumParams(EC=-1.0, flux_period=0.0, flux_bias=0.0),
        flux_unit="A",
        source=PredictorSource(None, None),
    )


@pytest.mark.parametrize("model", ["fluxonium", "transmon", "unrecognized"])
@pytest.mark.parametrize("empty_params", [False, True])
def test_undefined_explicit_model_contract_remains_located_pending(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
    legacy_fit: JsonObject,
    model: str,
    empty_params: bool,
) -> None:
    legacy_fit["model"] = model
    if empty_params:
        legacy_fit["params"] = {}
    source = write_params(request_data, {"fluxdep_fit": legacy_fit})
    report = migrate(request_data, analysis_mapping)
    assert any(
        item.source == source
        and item.location == "fluxdep_fit.model"
        and "legacy" in item.reason.lower()
        for item in report.pending
    )
    assert document_value(
        snapshot(open_entry(request_data).setup),
        "Qlegacy.ext.legacy_params.fluxdep_fit",
    ) == {
        key: value
        for key, value in legacy_fit.items()
        if key not in ("plot_transitions", "timestamp")
    }
    entry = open_entry(request_data)
    dispersive = load_dispersive_file(entry.dispersive_path / "Qlegacy.json")
    assert dispersive.source.created_at == legacy_fit["timestamp"]
    assert dispersive.params.plot_transitions is not None
    assert dispersive.params.plot_transitions.model_extra == {"0.1": [1, 2]}


def test_undefined_transmon_parameter_shape_is_preserved_as_unsupported(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
) -> None:
    fit: JsonObject = {"params": {"EJmax": 12.0, "EC": 0.3, "d": 0.2}}
    source = write_params(request_data, {"fluxdep_fit": fit})
    report = migrate(request_data, analysis_mapping)
    assert not any(item.source == source for item in report.pending)
    assert any(
        row.old_file == source
        and row.action == "preserved"
        and row.source_keys == ("fluxdep_fit", "params", "EJmax")
        and "unsupported model" in row.reason.lower()
        for row in report.key_mappings
    )
    assert not (request_data.results_root / request_data.name / "predictors/Qlegacy.json").exists()
    assert document_value(
        snapshot(open_entry(request_data).setup),
        "Qlegacy.ext.legacy_params.fluxdep_fit",
    ) == fit


@pytest.fixture
def interrupted_predictor(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
    legacy_fit: JsonObject,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    write_params(request_data, {"flux_unit": "mV", "fluxdep_fit": legacy_fit})
    write_meta(request_data, "bias", {})
    destination = request_data.results_root / request_data.name
    copyfile = shutil.copyfile

    def interrupted_copy(src: Path, dst: Path, *, follow_symlinks: bool = True) -> Path:
        if src == destination / "setup.yaml" and dst.name == "point.yaml":
            raise OSError("parameter copy interrupted")
        return copyfile(src, dst, follow_symlinks=follow_symlinks)

    with monkeypatch.context() as patch:
        patch.setattr(shutil, "copyfile", interrupted_copy)
        with pytest.raises(OSError, match="parameter copy interrupted"):
            migrate(request_data, analysis_mapping)
    predictor = destination / "predictors/Qlegacy.json"
    assert predictor.is_file(), "interruption must follow predictor publication"
    manifest = TypeAdapter(JsonObject).validate_json(
        manifest_path(request_data).read_text(encoding="utf-8")
    )
    assert manifest["parameters_complete"] is False
    return predictor


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", True),
        ("version", 1.0),
        ("flux_period", True),
        ("flux_bias", False),
        ("EJ", 9.0),
        ("params", {"unknown": 1}),
        ("model", "unknown"),
        ("source", {"run_id": "", "created_at": None}),
        ("source", {"run_id": None, "created_at": "2020-07-07T16:30:00"}),
        ("source", {"run_id": None}),
        ("source", {"run_id": None, "created_at": None, "extra": 1}),
        ("extra", 1),
    ],
)
def test_unfinished_resume_rejects_changed_or_schema_invalid_predictor(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
    interrupted_predictor: Path,
    field: str,
    value: JsonValue,
) -> None:
    document = read_predictor(request_data)
    if field in ("version", "params", "model", "source", "extra"):
        document[field] = value
    else:
        params = predictor_params(request_data)
        params[field] = value
        document["params"] = params
    interrupted_predictor.write_text(json.dumps(document), encoding="utf-8")
    before = interrupted_predictor.read_bytes()
    with pytest.raises(MigrationInputError, match="predictor"):
        migrate(replace(request_data, resume=True), analysis_mapping)
    assert interrupted_predictor.read_bytes() == before


@pytest.mark.parametrize(
    "legacy_fit",
    [{"params": {"EC": -1.0, "EL": None}, "flux_period": 0.0}],
)
@pytest.mark.parametrize("unknown_representation", ["omitted", "null"])
def test_unfinished_resume_compares_unknown_predictor_values_semantically(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
    interrupted_predictor: Path,
    unknown_representation: str,
) -> None:
    document = read_predictor(request_data)
    params = predictor_params(request_data)
    if unknown_representation == "omitted":
        params.pop("EL")
    else:
        params["EJ"] = None
    document["params"] = params
    interrupted_predictor.write_text(json.dumps(document), encoding="utf-8")
    before = interrupted_predictor.read_bytes()

    conflict = None
    try:
        migrate(replace(request_data, resume=True), analysis_mapping)
    except MigrationInputError as error:
        conflict = str(error)

    assert conflict is None, (
        "Unknown numeric omissions and nulls describe the same predictor"
    )
    assert interrupted_predictor.read_bytes() == before


def test_unfinished_resume_accepts_same_predictor_without_rewriting(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
    interrupted_predictor: Path,
) -> None:
    document = read_predictor(request_data)
    params = predictor_params(request_data)
    params["flux_period"] = 1
    document["params"] = params
    interrupted_predictor.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    before = interrupted_predictor.read_bytes()
    migrate(replace(request_data, resume=True), analysis_mapping)
    assert interrupted_predictor.read_bytes() == before
    document["flux_unit"] = "V"
    interrupted_predictor.write_text(json.dumps(document), encoding="utf-8")
    edited = interrupted_predictor.read_bytes()
    migrate(replace(request_data, resume=True), analysis_mapping)
    assert interrupted_predictor.read_bytes() == edited


def test_dry_run_predictor_plan_matches_apply_and_completed_resume_keeps_edits(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
    legacy_fit: JsonObject,
    tmp_path: Path,
) -> None:
    source = write_params(request_data, {"flux_unit": "mV", "fluxdep_fit": legacy_fit})
    before = {
        path: path.read_bytes() if path.is_file() else None
        for path in tmp_path.rglob("*")
    }
    dry = migrate(replace(request_data, dry_run=True), analysis_mapping)
    assert {
        path: path.read_bytes() if path.is_file() else None
        for path in tmp_path.rglob("*")
    } == before
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
