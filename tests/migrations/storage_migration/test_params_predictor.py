"""Historical fit projection and predictor evidence, publication and resume."""

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import JsonValue, TypeAdapter
from zcu_tools.datafile import JsonObject
from zcu_tools.resources.dispersive_file import load_dispersive_file
from zcu_tools.resources.entry import document_value, snapshot
from zcu_tools.resources.predictor_file import (
    PredictorFile,
    PredictorFlux,
    PredictorSource,
    load_predictor_file,
)
from zcu_tools.simulate.fluxonium.predict import FluxoniumParams

from migrations.storage_migration import (
    MigrationInputError,
    MigrationMapping,
    MigrationRequest,
)
from migrations.storage_migration.state import read_manifest
from tests.migrations.storage_migration.fakes import (
    assert_difference,
    manifest_path,
    write_meta,
)
from tests.migrations.storage_migration.params_support import (
    backup_path,
    load_migrated_predictor,
    migrate,
    open_entry,
    predictor_params,
    read_predictor,
    write_params,
)

pytestmark = pytest.mark.usefixtures("analysis_kind")


@pytest.mark.parametrize("missing", ["flux_half", "flux_period", "flux_bias"])
def test_incomplete_affine_stays_pending_without_publishing_predictor(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
    missing: str,
) -> None:
    fit: JsonObject = {
        "params": {"EJ": 6.0, "EC": 1.0, "EL": 0.8},
        "flux_half": 0.2,
        "flux_period": -2.0,
        "flux_bias": 0.05,
    }
    fit.pop(missing)
    source = write_params(request_data, {"fluxdep_fit": fit})
    report = migrate(request_data, analysis_mapping)
    assert any(
        item.source == source and missing in item.reason for item in report.pending
    )
    assert not (
        request_data.results_root / request_data.name / "predictors/Qlegacy.json"
    ).exists()
    assert (
        document_value(
            snapshot(open_entry(request_data).setup),
            "Qlegacy.ext.legacy_params.fluxdep_fit",
        )
        == fit
    )


@pytest.mark.parametrize("dry_run", [True, False])
@pytest.mark.parametrize(
    ("values", "missing"),
    [
        ({"half flux": 0.2, "period": -2.0}, "flux_bias"),
        ({"half flux": 0.2}, "flux_period"),
        ({"period": -2.0}, "flux_half"),
    ],
)
def test_partial_root_alignment_without_params_stays_pending_and_preserved(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
    dry_run: bool,
    values: JsonObject,
    missing: str,
) -> None:
    request = replace(request_data, dry_run=dry_run)
    source = write_params(request, values)
    original = source.read_bytes()
    before = {
        path: path.read_bytes()
        for path in request.result_root.parent.rglob("*")
        if path.is_file()
    }
    report = migrate(request, analysis_mapping)
    assert any(
        item.source == source and missing in item.reason for item in report.pending
    )
    assert not (
        request.results_root / request.name / "predictors/Qlegacy.json"
    ).exists()
    assert source.read_bytes() == original
    if dry_run:
        assert {
            path: path.read_bytes()
            for path in request.result_root.parent.rglob("*")
            if path.is_file()
        } == before
        assert not backup_path(request).exists()
    else:
        assert (
            document_value(
                snapshot(open_entry(request).setup), "Qlegacy.ext.legacy_params"
            )
            == values
        )
        assert (
            backup_path(request) / "result/chip/qubit/params.json"
        ).read_bytes() == original


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
        "period": -1.5,
        "flux_bias": 0.1,
        "name": "historical author",
        "project": {"name": "old project"},
        "schema_version": "legacy",
        "timestamp": "unproven local time",
    }
    source = write_params(request, values)
    before = {
        p: p.read_bytes() for p in request.result_root.parent.rglob("*") if p.is_file()
    }
    report = migrate(request, analysis_mapping)
    converted = {
        row.source_keys: row.new_path
        for row in report.key_mappings
        if row.old_file == source and row.action == "value"
    }
    assert converted.get(("params", "EJ")) == "params.EJ"
    assert converted.get(("params", "EC")) == "params.EC"
    assert converted.get(("params", "EL")) == "params.EL"
    assert converted.get(("half flux",)) == "flux"
    assert converted.get(("period",)) == "flux"
    assert any(item.location == "timestamp" for item in report.pending)
    assert not any(item.location == "flux_unit" for item in report.pending)
    preserved = {
        row.source_keys: row.destination_keys
        for row in report.key_mappings
        if row.action == "preserved"
    }
    assert preserved[("params", "custom.energy")] == (
        "Qlegacy",
        "ext",
        "legacy_params",
        "params",
        "custom.energy",
    )
    if dry_run:
        assert {
            p: p.read_bytes()
            for p in request.result_root.parent.rglob("*")
            if p.is_file()
        } == before
        assert not backup_path(request).exists()
        return
    predictor = load_migrated_predictor(request)
    assert predictor == PredictorFile(
        model="fluxonium",
        params=FluxoniumParams(EJ=6.27, EC=1.41, EL=0.8),
        flux=PredictorFlux(
            0.6, {"unnamed_device": -1.5}, {"unnamed_device": unit or "native"}
        ),
        source=PredictorSource(None, None),
    )
    assert document_value(
        snapshot(open_entry(request).setup), "Qlegacy.ext.legacy_params"
    ) == {
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
    legacy_fit["params"] = {
        energy_key: 0.8,
        "custom.energy": {"note": "untouched"},
        "EC": True,
    }
    source = write_params(request_data, {"fluxdep_fit": legacy_fit, "flux_unit": "mA"})
    report = migrate(request_data, analysis_mapping)
    predictor = (
        request_data.results_root / request_data.name / "predictors/Qlegacy.json"
    )
    assert predictor.is_file(), "an extra key must not block supported fluxonium data"
    assert load_predictor_file(predictor).params == FluxoniumParams.from_mapping(
        {energy_key: 0.8}
    )
    assert document_value(
        snapshot(open_entry(request_data).setup),
        "Qlegacy.ext.legacy_params.fluxdep_fit",
    ) == {
        "params": {"custom.energy": {"note": "untouched"}, "EC": True},
        "flux_int": -0.3,
    }
    assert any(item.location == "fluxdep_fit.params.EC" for item in report.pending)
    assert not any(item.location == "fluxdep_fit.params" for item in report.pending)
    assert any(
        row.old_file == source
        and row.action == "preserved"
        and row.source_keys == ("fluxdep_fit", "params", "custom.energy", "note")
        for row in report.key_mappings
    )


@pytest.mark.parametrize(
    "allows",
    [
        {},
        {
            "r_f": 5.2,
            "sample_f": 9.7,
            "transitions": [[0, 1]],
            "mirror": [[1, 2]],
            "author.note": "unchanged",
        },
    ],
)
def test_historical_allows_are_owner_annotations_without_reinterpretation(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
    allows: JsonObject,
) -> None:
    source = write_params(
        request_data,
        {
            "allows": allows,
            "timestamp": "2020-07-07T16:30:00+08:00",
            "name": "old author",
        },
    )
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
    assert document_value(snapshot(entry.setup), "Qlegacy.ext.legacy_params") == {
        "name": "old author"
    }
    assert any(
        row.old_file == source
        and row.action == "value"
        and row.source_keys == ("allows",)
        and row.new_path == "params.plot_transitions"
        for row in report.key_mappings
    )


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
        "params": {
            "EJ": 6.27 if equal else 8.2,
            "EC": 1.41,
            "EL": 0.8,
            "author": "keep",
        },
        "half flux": 0.2 if equal else 0.7,
        "period": 1.0,
    }
    source = write_params(request_data, values)
    report = migrate(request_data, analysis_mapping)
    assert load_migrated_predictor(request_data).params == FluxoniumParams(
        EJ=6.27, EC=1.41, EL=0.8
    )
    assert document_value(
        snapshot(open_entry(request_data).setup), "Qlegacy.ext.legacy_params"
    ) == {
        "fluxdep_fit": {"flux_int": -0.3},
        "params": {"author": "keep", **({} if equal else {"EJ": 8.2})},
        **({} if equal else {"half flux": 0.7}),
    }
    compared = [
        row
        for row in report.key_mappings
        if row.old_file == source and row.source_keys == ("params", "EJ")
    ]
    assert len(compared) == 1
    assert compared[0].action == ("remove" if equal else "preserved")
    assert compared[0].compared_source_keys == ("fluxdep_fit", "params", "EJ")
    assert "fluxdep_fit" in compared[0].reason
    saved = read_manifest(manifest_path(request_data)).report
    assert [
        row for row in saved.key_mappings if row.source_keys == ("params", "EJ")
    ] == compared


def test_newer_transition_annotations_preserve_only_different_historical_leaves(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
) -> None:
    source = write_params(
        request_data,
        {
            "allows": {
                "r_f": 5.2,
                "sample_f": 9.7,
                "transitions": [[0, 1]],
                "mirror": [[1, 2]],
                "nested": {"equal": [], "different": "old"},
                "old.only": "preserve",
            },
            "fluxdep_fit": {
                "plot_transitions": {
                    "r_f": 5.2,
                    "sample_f": 8.1,
                    "transitions": [[0, 1]],
                    "mirror": [[2, 3]],
                    "nested": {"equal": [], "different": "new"},
                }
            },
        },
    )
    report = migrate(request_data, analysis_mapping)
    entry = open_entry(request_data)
    assert load_dispersive_file(
        entry.dispersive_path / "Qlegacy.json"
    ).params.plot_transitions.model_dump(exclude_unset=True) == {
        "r_f": 5.2,
        "sample_f": 8.1,
        "transitions": [[0, 1]],
        "mirror": [[2, 3]],
        "nested": {"equal": [], "different": "new"},
    }
    assert document_value(snapshot(entry.setup), "Qlegacy.ext.legacy_params") == {
        "allows": {
            "sample_f": 9.7,
            "mirror": [[1, 2]],
            "nested": {"different": "old"},
            "old.only": "preserve",
        }
    }
    comparisons = {
        row.source_keys: row
        for row in report.key_mappings
        if row.old_file == source and row.compared_source_keys is not None
    }
    assert comparisons.get(("allows", "old.only")) is not None
    assert comparisons[("allows", "old.only")].action == "preserved"
    assert comparisons[("allows", "old.only")].compared_source_keys == (
        "fluxdep_fit",
        "plot_transitions",
        "old.only",
    )
    assert comparisons[("allows", "r_f")].action == "remove"
    assert comparisons[("allows", "nested", "equal")].action == "remove"
    different = comparisons[("allows", "nested", "different")]
    assert different.action == "preserved"
    assert different.compared_source_keys == (
        "fluxdep_fit",
        "plot_transitions",
        "nested",
        "different",
    )
    assert different.destination_keys == (
        "Qlegacy",
        "ext",
        "legacy_params",
        "allows",
        "nested",
        "different",
    )


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
    assert load_migrated_predictor(request).flux.unit == {"unnamed_device": "mA"}
    assert any(
        row.new_path == "flux.unit.unnamed_device" and "假設單位" in row.reason
        for row in report.key_mappings
    )
    assert not any(item.location == "fluxdep_fit.flux_unit" for item in report.pending)


@pytest.mark.parametrize(
    "values",
    [
        {"flux_unit": "V"},
        {"flux_unit": "A", "fluxdep_fit": {"flux_unit": "V"}},
        {"flux_unit": None},
        {"flux_unit": " "},
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
    report = migrate(request, analysis_mapping)
    if values.get("flux_unit") in ("V", None):
        assert load_migrated_predictor(request).flux.unit == {
            "unnamed_device": values.get("flux_unit") or "native"
        }
    else:
        assert report.pending
        assert not (
            request.results_root / request.name / "predictors/Qlegacy.json"
        ).exists()


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
        ),
        flux=PredictorFlux(
            0.3,
            {"unnamed_device": 1.0},
            {"unnamed_device": root_unit or fit_unit or "native"},
        ),
        source=PredictorSource(None, "2020-07-07T16:30:00+08:00"),
    )
    point = snapshot(open_entry(request_data).use_point("bias"))
    assert document_value(point, "Qlegacy.EJ") == 11.0
    assert document_value(point, "general.ext.flux_unit") == "mV"
    assert document_value(point, "Qlegacy.ext.legacy_params.fluxdep_fit") == {
        "flux_int": -0.3
    }


@pytest.mark.parametrize(
    "units", [{}, {"flux_unit": "A", "fit_unit": "V"}, {"flux_unit": " "}]
)
def test_missing_units_use_native_but_explicit_conflicts_remain_pending(
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
    if not units:
        assert load_migrated_predictor(request_data).flux.unit == {
            "unnamed_device": "native"
        }
    else:
        assert not (
            request_data.results_root / request_data.name / "predictors/Qlegacy.json"
        ).exists()
        assert any(
            item.source == source and "flux_unit" in item.reason
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


def test_invalid_affine_preserves_partial_energies_without_publishing_predictor(
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
    report = migrate(request_data, analysis_mapping)
    assert report.pending
    assert document_value(
        snapshot(open_entry(request_data).setup),
        "Qlegacy.ext.legacy_params.fluxdep_fit",
    ) == {
        "params": {"EC": -1.0, "EL": None},
        "flux_period": 0.0,
    }
    assert not (
        request_data.results_root / request_data.name / "predictors/Qlegacy.json"
    ).exists()


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
    assert not (
        request_data.results_root / request_data.name / "predictors/Qlegacy.json"
    ).exists()
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
        (
            "flux",
            {
                "offset": 0.3,
                "period": {"unnamed_device": True},
                "unit": {"unnamed_device": "mV"},
            },
        ),
        (
            "flux",
            {
                "offset": False,
                "period": {"unnamed_device": 1.0},
                "unit": {"unnamed_device": "mV"},
            },
        ),
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
    if field in ("version", "params", "model", "flux", "source", "extra"):
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
    [
        {
            "params": {"EC": -1.0, "EL": None},
            "flux_half": 0.2,
            "flux_period": -1.0,
            "flux_bias": 0.0,
        }
    ],
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
    params["EC"] = 1.41
    document["params"] = params
    interrupted_predictor.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    before = interrupted_predictor.read_bytes()
    migrate(replace(request_data, resume=True), analysis_mapping)
    assert interrupted_predictor.read_bytes() == before
    document["flux"] = {
        "offset": 0.3,
        "period": {"unnamed_device": 1.0},
        "unit": {"unnamed_device": "V"},
    }
    interrupted_predictor.write_text(json.dumps(document), encoding="utf-8")
    edited = interrupted_predictor.read_bytes()
    assert_difference(
        lambda: migrate(replace(request_data, resume=True), analysis_mapping),
        request_data,
        [interrupted_predictor],
    )
    assert interrupted_predictor.read_bytes() == edited


def test_dry_run_predictor_plan_matches_apply_and_completed_resume_reports_edits(
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
    edited["flux"] = {
        "offset": 0.3,
        "period": {"unnamed_device": 1.0},
        "unit": {"unnamed_device": "V"},
    }
    predictor_path.write_text(json.dumps(edited), encoding="utf-8")
    before = predictor_path.read_bytes()
    assert_difference(
        lambda: migrate(replace(request_data, resume=True), analysis_mapping),
        request_data,
        [predictor_path],
    )
    assert predictor_path.read_bytes() == before
