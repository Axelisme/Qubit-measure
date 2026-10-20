"""Literal metadata and raw params preservation without duplicating formal values."""

import json
from dataclasses import replace

import pytest
from pydantic import TypeAdapter
from zcu_tools.datafile import JsonObject
from zcu_tools.resources.dispersive_file import load_dispersive_file
from zcu_tools.resources.entry import document_value, snapshot
from zcu_tools.resources.predictor_file import load_predictor_file
from zcu_tools.resources.t1_curve_file import load_t1_curve_file
from zcu_tools.simulate.fluxonium.predict import FluxoniumParams

from migrations.storage_migration import (
    MigrationMapping,
    MigrationReport,
    MigrationRequest,
    report_json,
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
    write_params,
)

pytestmark = pytest.mark.usefixtures("analysis_kind")


@pytest.mark.parametrize("context", ["", "exps/bias"])
def test_unknown_meta_has_literal_preservation_rows_and_survives_resume(
    request_data: MigrationRequest,
    metadata_mapping: MigrationMapping,
    context: str,
) -> None:
    values: JsonObject = {
        "scalar": 42,
        "nested.with.dot": {"child": [1, None, {"x": True}]},
        "list": ["raw", 3],
        "null": None,
        "legacy_meta_info": {"source": "meta"},
        "legacy_module_cfg": ["source", "module"],
    }
    source = write_meta(request_data, context, values)
    root = request_data.result_root.parent
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    dry = migrate(replace(request_data, dry_run=True), metadata_mapping)
    assert {p: p.read_bytes() for p in root.rglob("*") if p.is_file()} == before
    report = migrate(request_data, metadata_mapping)
    entry = open_entry(request_data)
    view = entry.use_point("bias") if context else entry.setup
    target = entry.result_path / ("points/bias/point.yaml" if context else "setup.yaml")
    assert not [item for item in report.pending if item.source == source]
    ext = document_value(snapshot(view), "general.ext")
    assert ext == {
        "scalar": 42,
        "nested.with.dot": {"child": [1, None, {"x": True}]},
        "list": ["raw", 3],
        "null": None,
        "legacy_meta_info": {
            "legacy_meta_info": {"source": "meta"},
            "legacy_module_cfg": ["source", "module"],
        },
    }
    rows = [row for row in report.key_mappings if row.old_file == source]
    assert len(rows) == len(values)
    for row in rows:
        assert row.action == "preserved"
        assert row.source_keys == (row.old_key,)
        assert row.new_file == target
        assert row.new_path is None
        assert row.destination_keys == (
            ("general", "ext", "legacy_meta_info", row.old_key)
            if row.old_key in ("legacy_meta_info", "legacy_module_cfg")
            else ("general", "ext", row.old_key)
        )
    assert dry.key_mappings == report.key_mappings
    with view.edit() as draft:
        draft.set("general.description", "edited after migration")
    edited = target.read_bytes()
    assert_difference(
        lambda: migrate(replace(request_data, resume=True), metadata_mapping),
        request_data,
        [target],
    )
    assert target.read_bytes() == edited


@pytest.mark.parametrize(
    ("values", "params"),
    [
        ({"flx_half": 0.2, "best_jpa_flux": 0.3}, {}),
        ({"flux_unit": "V", "flx_half": 0.2}, {"flux_unit": "A"}),
        ({"flux_unit": "unknown", "flx_half": 0.2}, {}),
        ({"flux_unit": "A", "flx_half": "wrong type"}, {}),
        ({"jpa_flux_unit": "mA", "best_jpa_flux": 0.3}, {}),
        ({"jpa_flux_unit": None, "best_jpa_flux": 0.3}, {}),
    ],
)
def test_blocked_flux_preserves_actual_source_without_activating_units(
    request_data: MigrationRequest,
    metadata_mapping: MigrationMapping,
    values: JsonObject,
    params: JsonObject,
) -> None:
    if params:
        write_params(request_data, params)
    source = write_meta(request_data, "", {**values, "qub_ch": 4})
    dry = migrate(replace(request_data, dry_run=True), metadata_mapping)
    report = migrate(request_data, metadata_mapping)
    setup = snapshot(open_entry(request_data).setup)
    assert document_value(setup, "general.ext") == {"legacy_meta_info": values}
    assert document_value(setup, "Q1.wiring.ch") == 4
    view = open_entry(request_data).setup
    assert getattr(view.Q1, "flux_half", None) is None
    assert getattr(view.J1, "flux_unit", None) is None
    assert getattr(view.J1, "flux", None) is None
    pending = [item for item in report.pending if item.source == source]
    assert set(values) <= {item.location for item in pending}
    assert all("legacy_meta_info" in item.reason for item in pending)
    for key in values:
        row = next(
            row
            for row in report.key_mappings
            if row.old_file == source and row.old_key == key
        )
        assert row.action == "preserved"
        assert row.new_path is None
        assert row.source_keys == (key,)
        assert row.destination_keys == ("general", "ext", "legacy_meta_info", key)
    assert dry.key_mappings == report.key_mappings


@pytest.mark.parametrize("same", [True, False])
def test_qubit_channel_alias_is_deduplicated_only_against_selected_wiring(
    request_data: MigrationRequest, metadata_mapping: MigrationMapping, same: bool
) -> None:
    source = write_meta(request_data, "", {"qub_ch": 2 if same else 4, "qub_0_1_ch": 2})
    report = migrate(request_data, metadata_mapping)
    setup = snapshot(open_entry(request_data).setup)
    assert document_value(setup, "Q1.wiring.ch") == 2
    assert document_value(setup, "general.ext") == (
        {} if same else {"legacy_meta_info": {"qub_ch": 4}}
    )
    row = next(
        row
        for row in report.key_mappings
        if row.old_file == source and row.old_key == "qub_ch"
    )
    assert row.source_keys == ("qub_ch",)
    assert row.action == ("remove" if same else "preserved")
    if same:
        assert row.new_file == open_entry(request_data).result_path / "setup.yaml"
        assert row.new_path == "Q1.wiring.ch"
        assert "same" in row.reason.lower()
    else:
        assert row.destination_keys == ("general", "ext", "legacy_meta_info", "qub_ch")


@pytest.mark.parametrize("canonical", [None, 1.0, 2.0])
def test_removed_meta_keys_preserve_unmatched_values_and_report_proven_duplicates(
    request_data: MigrationRequest,
    metadata_mapping: MigrationMapping,
    canonical: float | None,
) -> None:
    removed: JsonObject = {
        "readout_f": 1.0,
        "flx_bias": 1.0,
        "mA_c": 1.0,
        "mA_e": 1.0,
        "resetf1_w": 1.0,
        "cur_jpa_A": 1.0,
        "fake_peak": {"value": 1.0},
        "log_scale": True,
    }
    values: JsonObject = {**removed, "flx_half": 1.0, "flux_unit": "V"}
    if canonical is not None:
        values.update({"r_f": canonical, "flux_bias": canonical})
    source = write_meta(request_data, "", values)
    report = migrate(request_data, metadata_mapping)
    expected = dict(removed)
    if canonical == 1.0:
        expected.pop("readout_f")
        expected.pop("flx_bias")
    assert document_value(snapshot(open_entry(request_data).setup), "general.ext") == {
        "flux_unit": "V",
        "legacy_meta_info": expected,
    }
    rows = {row.old_key: row for row in report.key_mappings if row.old_file == source}
    for key in removed:
        row = rows[key]
        assert row.source_keys == (key,)
        if canonical == 1.0 and key in ("readout_f", "flx_bias"):
            assert row.action == "remove"
            assert (
                row.new_path
                == {"readout_f": "R1.freq", "flx_bias": "Q1.ext.flux_bias"}[key]
            )
            assert row.new_file == open_entry(request_data).result_path / "setup.yaml"
            assert "same" in row.reason.lower()
        else:
            assert row.action == "preserved"
            assert row.destination_keys == ("general", "ext", "legacy_meta_info", key)
    assert not [item for item in report.pending if item.source == source]


def test_removed_alias_does_not_deduplicate_against_inherited_formal_value(
    request_data: MigrationRequest, metadata_mapping: MigrationMapping
) -> None:
    write_meta(request_data, "", {"r_f": 1.0, "flux_unit": "V", "flux_bias": 1.0})
    write_meta(request_data, "exps/bias", {"readout_f": 1.0, "flx_bias": 1.0})
    migrate(request_data, metadata_mapping)
    point = snapshot(open_entry(request_data).use_point("bias"))
    assert document_value(point, "general.ext") == {
        "flux_unit": "V",
        "legacy_meta_info": {"readout_f": 1.0, "flx_bias": 1.0},
    }


@pytest.mark.parametrize("raw_value", [0.4, 0.9])
@pytest.mark.parametrize("context", ["", "exps/bias"])
def test_unknown_meta_collision_keeps_formal_value_and_reports_exact_destination(
    request_data: MigrationRequest,
    metadata_mapping: MigrationMapping,
    raw_value: float,
    context: str,
) -> None:
    source = write_meta(
        request_data,
        context,
        {"cur_A": 0.4, "flux_unit": "V", "flux_value": raw_value},
    )
    dry = migrate(replace(request_data, dry_run=True), metadata_mapping)
    report = migrate(request_data, metadata_mapping)
    entry = open_entry(request_data)
    view = entry.use_point("bias") if context else entry.setup
    target = entry.result_path / ("points/bias/point.yaml" if context else "setup.yaml")
    assert document_value(snapshot(view), "general.ext.flux_value") == 0.4
    row = next(
        row
        for row in report.key_mappings
        if row.old_file == source and row.old_key == "flux_value"
    )
    assert row.source_keys == ("flux_value",)
    assert row.new_file == target
    if raw_value == 0.4:
        assert row.action == "remove"
        assert row.new_path == "general.ext.flux_value"
        assert "same" in row.reason.lower()
    else:
        assert document_value(snapshot(view), "general.ext.legacy_meta_info") == {
            "flux_value": 0.9
        }
        assert row.action == "preserved"
        assert row.new_path is None
        assert row.destination_keys == (
            "general",
            "ext",
            "legacy_meta_info",
            "flux_value",
        )
    assert not [item for item in report.pending if item.source == source]
    assert dry.key_mappings == report.key_mappings


def test_unknown_expression_collision_preserves_pending_and_formal_value(
    request_data: MigrationRequest, metadata_mapping: MigrationMapping
) -> None:
    source = write_meta(
        request_data,
        "",
        {"cur_A": 0.4, "flux_unit": "V", "flux_value": "=md.other"},
    )
    report = migrate(request_data, metadata_mapping)
    setup = snapshot(open_entry(request_data).setup)
    assert document_value(setup, "general.ext.flux_value") == 0.4
    assert document_value(setup, "general.ext.legacy_meta_info") == {
        "flux_value": "=md.other"
    }
    row = next(
        row
        for row in report.key_mappings
        if row.old_file == source and row.old_key == "flux_value"
    )
    assert row.action == "preserved"
    assert row.source_keys == ("flux_value",)
    assert row.destination_keys == ("general", "ext", "legacy_meta_info", "flux_value")
    assert row.new_path is None
    assert any(
        item.source == source
        and item.location == "flux_value"
        and "expression" in item.reason
        for item in report.pending
    )


@pytest.mark.parametrize("raw_value", [0.4, 0.9])
def test_point_unknown_collision_protects_only_successful_setup_projection(
    request_data: MigrationRequest,
    metadata_mapping: MigrationMapping,
    raw_value: float,
) -> None:
    write_meta(request_data, "", {"cur_A": 0.4, "flux_unit": "V", "arbitrary": 0.4})
    source = write_meta(
        request_data, "exps/bias", {"flux_value": raw_value, "arbitrary": raw_value}
    )
    dry = migrate(replace(request_data, dry_run=True), metadata_mapping)
    report = migrate(request_data, metadata_mapping)
    entry = open_entry(request_data)
    setup = snapshot(entry.setup)
    point = snapshot(entry.use_point("bias"))
    assert document_value(setup, "general.ext.flux_value") == 0.4
    assert document_value(point, "general.ext.flux_value") == 0.4
    assert document_value(point, "general.ext.arbitrary") == raw_value
    rows = {row.old_key: row for row in report.key_mappings if row.old_file == source}
    assert rows["arbitrary"].action == "preserved"
    assert rows["arbitrary"].destination_keys == ("general", "ext", "arbitrary")
    row = rows["flux_value"]
    assert row.source_keys == ("flux_value",)
    assert row.new_file == entry.result_path / "points/bias/point.yaml"
    if raw_value == 0.4:
        assert row.action == "remove"
        assert row.new_path == "general.ext.flux_value"
        assert "inherited" in row.reason.lower()
    else:
        assert document_value(point, "general.ext.legacy_meta_info") == {
            "flux_value": 0.9
        }
        assert row.action == "preserved"
        assert row.destination_keys == (
            "general",
            "ext",
            "legacy_meta_info",
            "flux_value",
        )
    assert not [item for item in report.pending if item.source == source]
    assert dry.key_mappings == report.key_mappings


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
    assert_difference(
        lambda: migrate(replace(request_data, resume=True), analysis_mapping),
        request_data,
        paths,
    )
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


def test_completed_resume_reports_user_edits_without_overwriting(
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
    assert_difference(
        lambda: migrate(replace(request_data, resume=True), analysis_mapping),
        request_data,
        paths,
    )
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
