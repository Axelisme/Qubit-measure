"""Public supplemental acquisition evidence loading contract."""

import json
from pathlib import Path

import pytest
from pydantic import JsonValue
from zcu_tools.datafile import JsonObject
from zcu_tools.resources.storage_migration import MigrationInputError, load_run_evidence


def evidence_document(*, entry_update: JsonObject | None = None) -> JsonObject:
    """A synthetic stopped acquisition with explicit historical metadata."""
    entry: JsonObject = {
        "source": "2025/10/scan.hdf5",
        "source_hash": "a" * 64,
        "experiment": "synthetic_scan",
        "cfg": {
            "values": {"frequency": 12.5, "future_cfg": {"retained": True}},
            "cfg_type": "SyntheticCfg",
            "schema_version": "1.0",
        },
        "started_at": "2025-10-01T00:00:00Z",
        "finished_at": None,
        "completion": "stopped",
        "snapshot": {
            "entry_name": "historical_name",
            "point": "historical_point",
            "description": None,
            "roles": {"probe": "C1"},
            "params": {
                "C1.frequency": {
                    "value": 12.5,
                    "unit": "MHz",
                    "source": {
                        "source": "manual",
                        "kind": None,
                        "run_id": None,
                        "at": "2025-09-30T00:00:00Z",
                        "stderr": 0.2,
                        "cloned_from": None,
                    },
                },
            },
            "future_snapshot": {"unknown": [1, 2, 3]},
        },
        "provenance": {
            "software_versions": {},
            "git_commit": None,
            "git_dirty": None,
            "qick_version": None,
            "soc_fingerprint": None,
            "hostname": None,
        },
        "future_entry": {"detail": "keep verbatim"},
    }
    if entry_update is not None:
        entry.update(entry_update)
    return {
        "format": "zcu.migration-run-evidence",
        "format_version": "1.7",
        "entries": [entry],
        "future_root": {"detail": [True, None, "retained"]},
    }


def write_document(tmp_path: Path, document: JsonObject) -> Path:
    source = tmp_path / "evidence.json"
    source.write_text(json.dumps(document), encoding="utf-8")
    return source


def test_load_preserves_future_fields_and_historical_snapshot(tmp_path: Path) -> None:
    document = evidence_document()
    source = write_document(tmp_path, document)

    loaded = load_run_evidence(source)

    assert loaded.raw == document
    assert source.read_text(encoding="utf-8") == json.dumps(document)
    entry = loaded.entries[0]
    assert entry.source == Path("2025/10/scan.hdf5")
    assert entry.experiment == "synthetic_scan"
    assert entry.completion == "stopped"
    assert entry.finished_at is None
    assert entry.snapshot.entry_name == "historical_name"
    assert entry.snapshot.point == "historical_point"
    assert entry.snapshot.roles == {"probe": "C1"}
    parameter = entry.snapshot.params["C1.frequency"]
    assert parameter.value == 12.5
    assert parameter.unit == "MHz"
    assert parameter.source.stderr == 0.2
    assert entry.cfg.values["future_cfg"] == {"retained": True}
    assert entry.provenance.hostname is None


def test_explicit_no_point_remains_none(tmp_path: Path) -> None:
    document = evidence_document(
        entry_update={
            "snapshot": {
                "entry_name": "no_point",
                "point": None,
                "description": None,
                "roles": {},
                "params": {},
            },
        },
    )

    loaded = load_run_evidence(write_document(tmp_path, document))

    assert loaded.entries[0].snapshot.point is None


@pytest.mark.parametrize("version", ["2.0", "0.9", "01.0", "1", "1.-1"])
def test_invalid_version_locates_document(tmp_path: Path, version: str) -> None:
    document = evidence_document()
    document["format_version"] = version
    source = write_document(tmp_path, document)

    with pytest.raises(MigrationInputError, match="format_version") as error:
        load_run_evidence(source)

    assert str(source) in str(error.value)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source", "/absolute.hdf5", "source"),
        ("source", "../escape.hdf5", "source"),
        ("source", "safe/../../escape.hdf5", "source"),
        ("source", "", "source"),
        ("source", ".", "source"),
        ("source_hash", "a" * 63, "source_hash"),
        ("source_hash", "A" * 64, "source_hash"),
        ("experiment", "", "experiment"),
        ("completion", "unknown", "completion"),
        ("completion", True, "completion"),
        ("started_at", "2025-10-01T00:00:00", "started_at"),
        ("started_at", "2025-10-01T00:00:00+08:00", "started_at"),
        ("started_at", "file_mtime", "started_at"),
        ("finished_at", "2025-09-30T00:00:00Z", "finished_at"),
        ("snapshot", {}, "snapshot"),
        ("cfg", {"values": []}, "cfg"),
    ],
)
def test_invalid_entry_is_not_silently_repaired(
    tmp_path: Path, field: str, value: JsonValue, message: str
) -> None:
    source = write_document(tmp_path, evidence_document(entry_update={field: value}))

    with pytest.raises(MigrationInputError, match=message) as error:
        load_run_evidence(source)

    assert str(source) in str(error.value)


def test_duplicate_source_is_rejected(tmp_path: Path) -> None:
    document = evidence_document()
    entries = document["entries"]
    assert isinstance(entries, list)
    document["entries"] = entries + entries
    source = write_document(tmp_path, document)

    with pytest.raises(MigrationInputError, match="duplicate"):
        load_run_evidence(source)


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "[]",
        '{"format":"other","format_version":"1.0","entries":[]}',
        '{"format":"zcu.migration-run-evidence","format_version":"1.0"}',
        '{"format":"zcu.migration-run-evidence","format_version":"1.0",'
        '"entries":[],"unknown":NaN}',
        '{"format":"zcu.migration-run-evidence","format_version":"1.0",'
        '"entries":[],"unknown":1e999}',
    ],
)
def test_malformed_document_reports_source(tmp_path: Path, text: str) -> None:
    source = tmp_path / "malformed.json"
    source.write_text(text, encoding="utf-8")

    with pytest.raises(MigrationInputError, match="malformed.json"):
        load_run_evidence(source)


def test_missing_file_propagates_io_failure(tmp_path: Path) -> None:
    source = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError, match="missing.json"):
        load_run_evidence(source)
