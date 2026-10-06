"""Distinct cfg identities under one tag and missing-evidence preservation."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from pydantic import TypeAdapter
from zcu_tools.datafile import (
    AxisSchema,
    CfgSnapshot,
    DataVariable,
    ExperimentPayload,
    LabberMetadata,
    LabberPayload,
    VariableSchema,
    load_run_data,
    write_labber,
)
from zcu_tools.resources.storage_migration import (
    LegacyRunEvidence,
    MigrationInputError,
    MigrationMapping,
    MigrationRequest,
    load_run_evidence,
    migrate_storage,
)

from tests.resources.storage_migration.fakes import (
    manifest_path,
    noop_validation,
    validate_cfg,
    write_run,
)

pytestmark = pytest.mark.usefixtures("registry_guard")


def test_parameters_do_not_validate_acquisition_cfg(
    request_data: MigrationRequest, mapping: MigrationMapping
) -> None:
    request, source = write_run(request_data, cfg_values={})

    def forbidden(tag: str, snapshot: CfgSnapshot) -> None:
        pytest.fail("Parameters-only must not validate run cfg")

    report = migrate_storage(
        replace(request, part="parameters"),
        mapping=mapping,
        validate_cfg=forbidden,
        validate_native=noop_validation,
    )
    assert source.is_file()
    assert report.converted_files == report.moved_labber_files == ()


def test_cfg_execution_error_propagates_instead_of_pending(
    request_data: MigrationRequest, mapping: MigrationMapping
) -> None:
    request, source = write_run(request_data)
    before = source.read_bytes()

    def unavailable(tag: str, snapshot: CfgSnapshot) -> None:
        validate_cfg(tag, snapshot)
        raise OSError("cfg owner unavailable")

    with pytest.raises(OSError, match="cfg owner unavailable"):
        migrate_storage(
            replace(request, part="data"),
            mapping=mapping,
            validate_cfg=unavailable,
            validate_native=noop_validation,
        )
    saved = json.loads(manifest_path(request).read_text())
    assert saved["runs"] == []
    assert saved["report"]["pending"] == []
    assert source.read_bytes() == before


@pytest.mark.parametrize("flag, replacement", [(True, 1), (False, 0)])
def test_typed_raw_boolean_number_disagreement_is_rejected(
    request_data: MigrationRequest,
    mapping: MigrationMapping,
    flag: bool,
    replacement: int,
) -> None:
    request, source = write_run(
        request_data, cfg_values={"frequency": 12.5, "nested": [{"flag": flag}]}
    )
    evidence = request.run_evidence
    assert evidence is not None
    original = evidence.entries[0]
    assert original.cfg is not None
    changed = replace(
        original,
        cfg=replace(
            original.cfg, values={"frequency": 12.5, "nested": [{"flag": replacement}]}
        ),
    )
    before = source.read_bytes()
    with pytest.raises(MigrationInputError, match="typed projection"):
        migrate_storage(
            replace(request, run_evidence=replace(evidence, entries=(changed,))),
            mapping=mapping,
            validate_cfg=validate_cfg,
            validate_native=noop_validation,
        )
    assert source.read_bytes() == before
    assert original.cfg.values["nested"] == [{"flag": flag}]


@pytest.mark.parametrize("flag, replacement", [(True, 1), (False, 0)])
def test_cfg_boolean_number_conflict_retains_source(
    request_data: MigrationRequest,
    mapping: MigrationMapping,
    flag: bool,
    replacement: int,
) -> None:
    request, source = write_run(
        request_data, cfg_values={"frequency": 12.5, "nested": [{"flag": flag}]}
    )
    assert request.run_evidence is not None
    raw = json.loads(json.dumps(request.run_evidence.raw))
    raw["entries"][0]["cfg"]["values"]["nested"][0]["flag"] = replacement
    evidence_path = request.result_root.parent / "boolean-conflict.json"
    evidence_path.write_text(json.dumps(raw))
    before = source.read_bytes()
    report = migrate_storage(
        replace(request, run_evidence=load_run_evidence(evidence_path)),
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    assert report.converted_files == report.moved_labber_files == ()
    assert any("comment.cfg" in item.reason for item in report.pending)
    assert source.read_bytes() == before


@pytest.mark.parametrize("flag, replacement", [(True, 1), (False, 0)])
def test_planned_raw_boolean_number_replacement_is_refused(
    request_data: MigrationRequest,
    mapping: MigrationMapping,
    flag: bool,
    replacement: int,
) -> None:
    request, source = write_run(request_data)
    assert request.run_evidence is not None
    raw = json.loads(json.dumps(request.run_evidence.raw))
    raw["entries"][0]["future_nested"] = {"flags": [flag]}
    evidence_path = request.result_root.parent / "planned-raw.json"
    evidence_path.write_text(json.dumps(raw))
    request = replace(request, run_evidence=load_run_evidence(evidence_path))
    report = migrate_storage(
        request,
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    before = {
        item.destination: item.destination.read_bytes()
        for item in (*report.converted_files, *report.moved_labber_files)
    }
    raw["entries"][0]["future_nested"]["flags"][0] = replacement
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
    saved = json.loads(manifest_path(request).read_text())
    assert saved["evidence"]["entries"][0]["future_nested"]["flags"][0] is flag
    assert {path: path.read_bytes() for path in before} == before
    assert not source.exists()


@pytest.mark.parametrize("dry_run", [False, True])
def test_invalid_cfg_is_pending_before_run_assignment(
    request_data: MigrationRequest, mapping: MigrationMapping, dry_run: bool
) -> None:
    request, source = write_run(request_data, cfg_values={})
    before = source.read_bytes()
    report = migrate_storage(
        replace(request, part="data", dry_run=dry_run),
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    assert report.converted_files == report.moved_labber_files == ()
    assert any(
        item.source == source and "frequency" in item.reason for item in report.pending
    )
    assert source.read_bytes() == before
    if dry_run:
        assert not manifest_path(request).exists()
        assert not (request.database_root / request.name).exists()
        return
    manifest = json.loads(manifest_path(request).read_text())
    assert manifest["runs"] == []
    assert not any(item["operation"] == "native" for item in manifest["files"])
    assert request.run_evidence is not None
    assert manifest["evidence"] == request.run_evidence.raw


@pytest.mark.parametrize("cfg_type", [None, "", "UnregisteredCfg"])
def test_missing_or_undeclared_cfg_identity_is_pending_without_tag_fallback(
    request_data: MigrationRequest,
    mapping: MigrationMapping,
    cfg_type: str | None,
) -> None:
    request, source = write_run(request_data)
    assert request.run_evidence is not None
    document = request.run_evidence.raw
    entries = document["entries"]
    assert isinstance(entries, list) and isinstance(entries[0], dict)
    raw_cfg = entries[0]["cfg"]
    assert isinstance(raw_cfg, dict)
    if cfg_type is None:
        raw_cfg.pop("cfg_type")
    else:
        raw_cfg["cfg_type"] = cfg_type
    evidence_path = request.result_root.parent / "missing-cfg-identity.json"
    evidence_path.write_text(json.dumps(document), encoding="utf-8")
    evidence = load_run_evidence(evidence_path)
    assert evidence.raw == document

    def forbidden(path: Path, tag: str, cfg_type: str) -> None:
        pytest.fail("An unresolved identity must never reach validation")

    report = migrate_storage(
        replace(request, part="data", run_evidence=evidence),
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=forbidden,
    )
    assert source.is_file()
    assert report.converted_files == report.moved_labber_files == ()
    assert any("cfg_type" in item.reason for item in report.pending)


@pytest.mark.parametrize(
    ("identity", "native_tag"),
    [(("synthetic_scan", "SyntheticCfg"), ""), (("unknown", "SyntheticCfg"), "new")],
)
def test_invalid_native_tag_declaration_is_rejected_before_writes(
    request_data: MigrationRequest,
    mapping: MigrationMapping,
    identity: tuple[str, str],
    native_tag: str,
) -> None:
    request, source = write_run(request_data)
    original = source.read_bytes()
    with pytest.raises(MigrationInputError, match="native_tags"):
        migrate_storage(
            replace(request, part="data"),
            mapping=replace(mapping, native_tags={identity: native_tag}),
            validate_cfg=validate_cfg,
            validate_native=noop_validation,
        )
    assert source.read_bytes() == original
    assert not manifest_path(request).exists()
    assert not (request.results_root / request.name).exists()
    assert not (request.database_root / request.name).exists()


@pytest.mark.parametrize("native_tag", ["synthetic_scan", "synthetic_gain"])
def test_same_tag_selects_distinct_cfg_type_schemas_and_callbacks(
    request_data: MigrationRequest,
    mapping: MigrationMapping,
    native_tag: str,
) -> None:
    request, first_source = write_run(request_data)
    assert request.run_evidence is not None
    second_source = first_source.with_name("second.hdf5")
    cfg = CfgSnapshot(
        values={"other": 3}, cfg_type="AlternateCfg", schema_version="1.0"
    )
    write_labber(
        second_source,
        ExperimentPayload(
            variables={
                DataVariable("signal"): LabberPayload(
                    ("signal", "V", np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
                    [
                        ("y", "s", np.array([0.0, 1.0, 2.0])),
                        ("x", "s", np.array([0.0, 1.0])),
                    ],
                )
            },
            metadata=LabberMetadata(tags=["synthetic_scan"]),
            representation="single",
        ),
        cfg=cfg,
    )
    with second_source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    second = replace(
        request.run_evidence.entries[0],
        source=Path("2025/10/second.hdf5"),
        source_hash=digest,
        cfg=cfg,
    )
    entries = request.run_evidence.raw["entries"]
    assert isinstance(entries, list)
    entries.append(TypeAdapter(LegacyRunEvidence).dump_python(second, mode="json"))
    evidence_path = request.result_root.parent / "dual-schema-evidence.json"
    evidence_path.write_text(json.dumps(request.run_evidence.raw), encoding="utf-8")
    schemas = {
        **mapping.data_schemas,
        ("synthetic_scan", "AlternateCfg"): (
            VariableSchema(
                variable=DataVariable("signal"),
                axes=(
                    AxisSchema(name="y", unit="s", dtype=np.dtype("float64")),
                    AxisSchema(name="x", unit="s", dtype=np.dtype("float64")),
                ),
                signal_name="signal",
                signal_unit="V",
                signal_dtype=np.dtype("float64"),
            ),
        ),
    }
    observed: dict[str, tuple[int, ...]] = {}

    def validate(path: Path, tag: str, cfg_type: str) -> None:
        stored = load_run_data(path)
        assert tag == "synthetic_scan"
        expected_tag = native_tag if cfg_type == "AlternateCfg" else tag
        assert stored.metadata.experiment == expected_tag
        assert cfg_type == stored.cfg.cfg_type
        observed[cfg_type] = stored.payload.variables[DataVariable("signal")].z.shape

    report = migrate_storage(
        replace(request, part="data", run_evidence=load_run_evidence(evidence_path)),
        mapping=replace(
            mapping,
            data_schemas=schemas,
            native_tags={("synthetic_scan", "AlternateCfg"): native_tag},
        ),
        validate_cfg=validate_cfg,
        validate_native=validate,
    )
    assert observed == {"SyntheticCfg": (2,), "AlternateCfg": (2, 3)}
    assert len(report.converted_files) == len(report.moved_labber_files) == 2
    assert not first_source.exists() and not second_source.exists()
    raw_before = json.loads(manifest_path(request).read_text(encoding="utf-8"))
    evidence_before = raw_before["evidence"]
    assert evidence_before == json.loads(evidence_path.read_text(encoding="utf-8"))
    resumed = migrate_storage(
        replace(request, part="data", resume=True, run_evidence=None),
        mapping=replace(
            mapping,
            data_schemas=schemas,
            native_tags={("synthetic_scan", "AlternateCfg"): native_tag},
        ),
        validate_cfg=validate_cfg,
        validate_native=validate,
    )
    assert resumed.converted_files == report.converted_files
    raw_after = json.loads(manifest_path(request).read_text(encoding="utf-8"))
    assert raw_after["evidence"] == evidence_before
