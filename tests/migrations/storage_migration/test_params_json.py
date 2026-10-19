"""Legacy analysis handoff conversion through migrate_storage and ResultEntry.

Remove with the offline migrations package once legacy entries are converted.
"""

import importlib
import json
import shutil
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
    MigrationInputError,
    MigrationMapping,
    MigrationReport,
    MigrationRequest,
    migrate_storage,
)
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
    for key, value in analysis_sections.items():
        assert document_value(doc, f"Qlegacy.ext.legacy_params.{key}") == value


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
    source = write_params(request_data, analysis_sections)
    destination = request_data.results_root / request_data.name
    copyfile = shutil.copyfile

    def interrupt(src: Path, dst: Path, *, follow_symlinks: bool = True) -> Path:
        if src == source and dst.parent == destination:
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


@pytest.fixture
def interrupted_predictor(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
    legacy_fit: JsonObject,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    source = write_params(request_data, {"flux_unit": "mV", "fluxdep_fit": legacy_fit})
    destination = request_data.results_root / request_data.name
    copyfile = shutil.copyfile

    def interrupted_copy(src: Path, dst: Path, *, follow_symlinks: bool = True) -> Path:
        if src == source and dst.parent == destination:
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
    ],
)
def test_unfinished_resume_rejects_changed_or_schema_invalid_predictor(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
    interrupted_predictor: Path,
    field: str,
    value: bool | float,
) -> None:
    document = read_predictor(request_data)
    if field == "version":
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
    assert (request_data.results_root / request_data.name / "params.json").is_file()
    document["flux_unit"] = "V"
    interrupted_predictor.write_text(json.dumps(document), encoding="utf-8")
    edited = interrupted_predictor.read_bytes()
    migrate(replace(request_data, resume=True), analysis_mapping)
    assert interrupted_predictor.read_bytes() == edited


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
