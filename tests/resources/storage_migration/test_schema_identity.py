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
    MigrationMapping,
    MigrationRequest,
    load_run_evidence,
    migrate_storage,
)

from tests.resources.storage_migration.fakes import write_run

pytestmark = pytest.mark.usefixtures("registry_guard")


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
        validate_native=forbidden,
    )
    assert source.is_file()
    assert report.converted_files == report.moved_labber_files == ()
    assert any("cfg_type" in item.reason for item in report.pending)


def test_same_tag_selects_distinct_cfg_type_schemas_and_callbacks(
    request_data: MigrationRequest,
    mapping: MigrationMapping,
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
        assert tag == stored.metadata.experiment == "synthetic_scan"
        assert cfg_type == stored.cfg.cfg_type
        observed[cfg_type] = stored.payload.variables[DataVariable("signal")].z.shape

    report = migrate_storage(
        replace(request, part="data", run_evidence=load_run_evidence(evidence_path)),
        mapping=replace(mapping, data_schemas=schemas),
        validate_native=validate,
    )
    assert observed == {"SyntheticCfg": (2,), "AlternateCfg": (2, 3)}
    assert len(report.converted_files) == len(report.moved_labber_files) == 2
    assert not first_source.exists() and not second_source.exists()
