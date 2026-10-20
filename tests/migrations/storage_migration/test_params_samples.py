"""Sample declarations, lossless CSV publication and resumable runtime rows."""

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from migrations.storage_migration import (
    MigrationInputError,
    MigrationMapping,
    MigrationRequest,
    migrate_storage,
)
from tests.migrations.storage_migration.fakes import (
    manifest_path,
    noop_validation,
    validate_cfg,
)
from tests.migrations.storage_migration.params_support import (
    declare_request,
    migrate,
    open_entry,
)

pytestmark = pytest.mark.usefixtures("analysis_kind")


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
