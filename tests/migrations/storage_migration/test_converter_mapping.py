"""Explicit mapping and run evidence govern accepted values and preserved sources."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import TypeAdapter
from zcu_tools.datafile import JsonObject
from zcu_tools.resources.entry import ResultEntry, document_value, snapshot
from zcu_tools.resources.entry.views import FieldView

from migrations.storage_migration import (
    KeyRule,
    LegacyParameterSnapshot,
    LegacyParameterSource,
    LegacyRunEvidence,
    MigrationInputError,
    MigrationMapping,
    MigrationRequest,
    ModuleRule,
    load_run_evidence,
    migrate_storage,
)
from tests.migrations.storage_migration.containers import read_migrated
from tests.migrations.storage_migration.fakes import (
    assert_difference,
    manifest_path,
    noop_validation,
    validate_cfg,
    write_meta,
    write_run,
)

pytestmark = pytest.mark.usefixtures("registry_guard")


def test_two_complete_points_keep_working_units_and_sources(
    request_data: MigrationRequest, mapping: MigrationMapping
) -> None:
    source_a = write_meta(
        request_data,
        "context_a",
        {
            "old_error": 0.25,
            "old_frequency": 12.5,
            "unknown": {"legacy": 1},
            "expression": "${md.old_frequency}",
            "obsolete": 999,
        },
    )
    source_b = write_meta(request_data, "context_b", {"old_frequency": 42.0})
    before = (source_a.read_bytes(), source_b.read_bytes())
    report = migrate_storage(
        replace(request_data, part="parameters"),
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    entry = ResultEntry.open(
        request_data.name,
        result_root=request_data.results_root,
        database_root=request_data.database_root,
    )
    assert entry.setup.C1.frequency == 1.0
    a, b = entry.use_point("context_a"), entry.use_point("context_b")
    assert a.C1.frequency == 12.5
    assert b.C1.frequency == 42.0
    assert a.meta("C1.frequency") == {"stderr": 0.25}
    unknown = a.general.ext["unknown"]
    assert isinstance(unknown, FieldView)
    assert unknown["legacy"] == 1
    assert a.general.ext["expression"] == "${md.old_frequency}"
    assert {item.location for item in report.pending} == {"expression"}
    unknown_row = next(row for row in report.key_mappings if row.old_key == "unknown")
    assert unknown_row.action == "preserved"
    assert unknown_row.old_file == source_a
    assert unknown_row.destination_keys == ("general", "ext", "unknown")
    assert (source_a.read_bytes(), source_b.read_bytes()) == before
    entry.setup.C1.frequency = 77.0
    assert entry.use_point("context_a").C1.frequency == 12.5


def test_data_resume_reports_valid_user_edit_without_overwriting(
    request_data: MigrationRequest, mapping: MigrationMapping
) -> None:
    write_meta(request_data, "context", {"old_frequency": 8.0})
    migrate_storage(
        replace(request_data, part="parameters"),
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    entry = ResultEntry.open(
        request_data.name,
        result_root=request_data.results_root,
        database_root=request_data.database_root,
    )
    entry.use_point("context").C1.frequency = 19.0
    assert_difference(
        lambda: migrate_storage(
            replace(request_data, part="data", resume=True),
            mapping=mapping,
            validate_cfg=validate_cfg,
            validate_native=noop_validation,
        ),
        request_data,
        [entry.result_path / "points/context/point.yaml"],
    )
    assert entry.use_point("context").C1.frequency == 19.0


def test_invalid_current_point_is_not_restored(
    request_data: MigrationRequest, mapping: MigrationMapping
) -> None:
    write_meta(request_data, "context", {"old_frequency": 8.0})
    migrate_storage(
        replace(request_data, part="parameters"),
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    point = (
        request_data.results_root
        / request_data.name
        / "points"
        / "context"
        / "point.yaml"
    )
    point.write_text("invalid: true")
    with pytest.raises(ValueError, match="format|general|validation"):
        migrate_storage(
            replace(request_data, part="data", resume=True),
            mapping=mapping,
            validate_cfg=validate_cfg,
            validate_native=noop_validation,
        )
    assert point.read_text() == "invalid: true"


@pytest.mark.parametrize("origin", ["manual", "missing-event", "invalid-time"])
def test_historical_parameters_remain_evidence_and_unresolved_sources_stay_pending(
    request_data: MigrationRequest, mapping: MigrationMapping, origin: str
) -> None:
    request, source = write_run(replace(request_data, part="data"))
    assert request.run_evidence is not None
    evidence = request.run_evidence.entries[0]
    parameter = LegacyParameterSnapshot(
        value=12.5,
        unit="MHz",
        source=LegacyParameterSource(
            source="manual" if origin == "invalid-time" else origin,
            kind=None,
            run_id=None,
            at="2025-09-30T00:00:00"
            if origin == "invalid-time"
            else "2025-09-30T00:00:00Z",
            stderr=0.2,
            cloned_from=None,
        ),
    )
    evidence = replace(
        evidence,
        snapshot=replace(evidence.snapshot, params={"C1.frequency": parameter}),
    )
    raw = json.loads(json.dumps(request.run_evidence.raw))
    raw["entries"][0] = TypeAdapter(LegacyRunEvidence).dump_python(
        evidence, mode="json"
    )
    path = request.result_root.parent / "with-parameters.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    request = replace(request, run_evidence=load_run_evidence(path))
    report = migrate_storage(
        request,
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    manifest = json.loads(manifest_path(request).read_text())
    assert manifest["evidence"] == raw
    if origin == "manual":
        assert len(report.converted_files) == 1
        stored = read_migrated(report.converted_files[0].destination)
        assert stored.cfg == evidence.cfg
        assert len(report.moved_labber_files) == 1
    else:
        assert source.is_file()
        assert not report.converted_files
        expected_reason = "timestamp" if origin == "invalid-time" else "missing-event"
        assert any(expected_reason in item.reason for item in report.pending)


def test_planned_evidence_is_immutable_including_future_fields(
    request_data: MigrationRequest, mapping: MigrationMapping
) -> None:
    request, source = write_run(request_data)

    def fail(path: Path, tag: str, cfg_type: str) -> None:
        noop_validation(path, tag, cfg_type)
        raise RuntimeError("pause validation")

    with pytest.raises(RuntimeError, match="pause"):
        migrate_storage(
            request, mapping=mapping, validate_cfg=validate_cfg, validate_native=fail
        )
    assert request.run_evidence is not None
    raw = json.loads(json.dumps(request.run_evidence.raw))
    raw["entries"][0]["future_detail"] = "changed evidence"
    evidence_path = request.result_root.parent / "changed-evidence.json"
    evidence_path.write_text(json.dumps(raw))
    with pytest.raises(MigrationInputError, match="planned source"):
        migrate_storage(
            replace(
                request, resume=True, run_evidence=load_run_evidence(evidence_path)
            ),
            mapping=mapping,
            validate_cfg=validate_cfg,
            validate_native=noop_validation,
        )
    assert source.is_file()


@pytest.mark.parametrize(
    "values",
    [{"old_error": 0.25}, {"old_error": 0.25, "old_frequency": "${md.other}"}],
)
def test_stderr_requires_a_value_accepted_by_this_mapping(
    request_data: MigrationRequest, mapping: MigrationMapping, values: JsonObject
) -> None:
    write_meta(request_data, "context", values)
    report = migrate_storage(
        replace(request_data, part="parameters"),
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    point = ResultEntry.open(
        request_data.name,
        result_root=request_data.results_root,
        database_root=request_data.database_root,
    ).use_point("context")
    assert point.C1.frequency == 1.0
    assert point.meta("C1.frequency") is None
    pending = [item for item in report.pending if item.location == "old_error"]
    assert [item.reason for item in pending] == [
        "No accepted explicit target value for stderr"
    ]


@pytest.mark.parametrize(
    "unit_evidence", [None, "declared-working-unit", "=unresolved"]
)
def test_parameter_requires_explicit_unit_evidence(
    request_data: MigrationRequest,
    mapping: MigrationMapping,
    unit_evidence: str | None,
) -> None:
    values: JsonObject = {"old_frequency": 12.5}
    if unit_evidence is not None:
        values["unit_evidence"] = unit_evidence
    write_meta(request_data, "context", values)
    rule = replace(mapping.rules[0], requires_keys=("unit_evidence",))
    report = migrate_storage(
        replace(request_data, part="parameters"),
        mapping=replace(mapping, rules=(rule,)),
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    point = ResultEntry.open(
        request_data.name,
        result_root=request_data.results_root,
        database_root=request_data.database_root,
    ).use_point("context")
    accepted = unit_evidence == "declared-working-unit"
    assert point.C1.frequency == (12.5 if accepted else 1.0)
    pending = [item for item in report.pending if item.location == "old_frequency"]
    assert bool(pending) is not accepted
    if pending:
        assert "unit_evidence" in pending[0].reason


def test_wrapped_channel_creates_a_complete_optional_container(
    request_data: MigrationRequest,
    mapping: MigrationMapping,
) -> None:
    write_meta(request_data, "context", {"old_channel": 7})
    report = migrate_storage(
        replace(request_data, part="parameters"),
        mapping=replace(
            mapping,
            rules=(
                KeyRule(
                    old_key="old_channel",
                    target_path="C1.channel",
                    wrap_key="ch",
                    action="value",
                    reason="Complete channel from explicit evidence",
                ),
            ),
        ),
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    point = ResultEntry.open(
        request_data.name,
        result_root=request_data.results_root,
        database_root=request_data.database_root,
    ).use_point("context")
    channel = point.C1.channel
    assert isinstance(channel, FieldView)
    assert channel.ch == 7
    decision = next(
        item for item in report.key_mappings if item.old_key == "old_channel"
    )
    assert decision.new_path == "C1.channel.ch"


def test_unsupported_module_candidates_preserve_source_without_slot(
    request_data: MigrationRequest,
    mapping: MigrationMapping,
) -> None:
    meta = write_meta(request_data, "context", {"old_frequency": 12.5})
    library = meta.parent / "module_cfg.yaml"
    library.write_text(
        "waveforms: {pulse: {style: const, length: 1}}\n"
        "modules:\n"
        "  second_choice: {type: synthetic, value: 2}\n"
        "  third_choice: {type: synthetic, value: 3}\n"
        "  first_choice: {type: synthetic, value: 1}\n"
        "  unknown_module: {type: unknown}\n"
        "future_root: {keep: true}\n",
        encoding="utf-8",
    )
    original = library.read_bytes()
    rules = tuple(
        ModuleRule(
            old_name=name,
            target_path=f"C1.pulses.{name}",
            reference_path="C1.module.chosen",
            reason="Explicit candidate",
        )
        for name in ("first_choice", "second_choice", "third_choice")
    )
    report = migrate_storage(
        replace(request_data, part="parameters"),
        mapping=replace(mapping, module_rules=rules),
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    point = ResultEntry.open(
        request_data.name,
        result_root=request_data.results_root,
        database_root=request_data.database_root,
    ).use_point("context")
    assert document_value(snapshot(point), "C1.module") == {}
    assert document_value(snapshot(point), "general.ext.legacy_module_cfg") == {
        "modules": {
            "second_choice": {"type": "synthetic", "value": 2},
            "third_choice": {"type": "synthetic", "value": 3},
            "first_choice": {"type": "synthetic", "value": 1},
            "unknown_module": {"type": "unknown"},
        },
        "waveforms": {"pulse": {"style": "const", "length": 1}},
        "future_root": {"keep": True},
    }
    for keys in (
        ("modules", "first_choice"),
        ("modules", "second_choice"),
        ("modules", "third_choice"),
        ("modules", "unknown_module"),
        ("waveforms", "pulse"),
        ("future_root",),
    ):
        assert any(
            item.old_file == library
            and item.action == "preserved"
            and item.source_keys == keys
            and item.destination_keys == ("general", "ext", "legacy_module_cfg", *keys)
            and item.new_file
            == request_data.results_root
            / request_data.name
            / "points/context/point.yaml"
            and item.new_path is None
            for item in report.key_mappings
        )
    assert not any(item.source == library for item in report.pending)
    assert library.read_bytes() == original
