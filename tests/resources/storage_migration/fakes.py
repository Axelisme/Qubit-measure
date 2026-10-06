"""Synthetic entry and acquisition builders shared by migration contract tests."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from zcu_tools.datafile import (
    CfgSnapshot,
    DataVariable,
    ExperimentPayload,
    JsonObject,
    LabberMetadata,
    LabberPayload,
    SoftwareProvenance,
    write_labber,
)
from zcu_tools.resources.entry import ComponentSchema
from zcu_tools.resources.storage_migration import (
    LegacyRunEvidence,
    LegacySnapshotEvidence,
    MigrationRequest,
    load_run_evidence,
)


class SyntheticChannel(BaseModel):
    """A complete test channel with one nonnegative integer ch."""

    ch: int = Field(ge=0, strict=True)


class SyntheticProbe(ComponentSchema):
    """Test-only component with working-unit fields and no concrete lab names.

    kind is migration_test_probe. frequency is a working-unit scalar. channel
    is an explicit SyntheticChannel or None. module maps test slots to string
    library references; no module cfg values live in this component.
    """

    model_config = ConfigDict(extra="forbid")
    kind: str = "migration_test_probe"
    frequency: float
    channel: SyntheticChannel | None = None
    module: dict[str, str] = Field(default_factory=dict)


def write_meta(request: MigrationRequest, label: str, values: JsonObject) -> Path:
    """Write UTF-8 meta_info.json under fixture chip/qubit/label and return its Path.

    request supplies the result root; values is the complete synthetic JSON.
    Create parents and replace this fixture file. I/O errors propagate.
    """
    source = request.result_root / "chip" / "qubit" / label / "meta_info.json"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(json.dumps(values), encoding="utf-8")
    return source


def write_run(request: MigrationRequest) -> tuple[MigrationRequest, Path]:
    """Write one canonical synthetic Labber run plus its acquisition evidence.

    request uses fixture chip/qubit roots. Return a replaced request carrying the
    full evidence document and the Labber Path. Use SyntheticCfg/synthetic_scan
    with a stopped historical snapshot. Create parents and replace fixture
    evidence JSON; writer/I/O errors propagate.
    """
    source = request.database_root / "chip" / "qubit" / "2025" / "10" / "scan.hdf5"
    source.parent.mkdir(parents=True, exist_ok=True)
    cfg = CfgSnapshot(
        values={"frequency": 12.5}, cfg_type="SyntheticCfg", schema_version="1.0"
    )
    payload = ExperimentPayload(
        variables={
            DataVariable("signal"): LabberPayload(
                ("signal", "V", np.array([1 + 2j, 3 + 4j])),
                [("x", "s", np.array([0.0, 1.0]))],
            )
        },
        metadata=LabberMetadata(tags=["synthetic_scan"]),
        representation="single",
    )
    write_labber(source, payload, cfg=cfg)
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    evidence = LegacyRunEvidence(
        source=Path("2025/10/scan.hdf5"),
        source_hash=digest,
        experiment="synthetic_scan",
        cfg=cfg,
        started_at="2025-10-01T00:00:00Z",
        finished_at=None,
        completion="stopped",
        snapshot=LegacySnapshotEvidence(
            entry_name="historical",
            point="point_at_acquisition",
            description=None,
            roles={},
            params={},
        ),
        provenance=SoftwareProvenance(
            software_versions={},
            git_commit=None,
            git_dirty=None,
            qick_version=None,
            soc_fingerprint=None,
            hostname=None,
        ),
    )
    raw = TypeAdapter(LegacyRunEvidence).dump_python(evidence, mode="json")
    document = {
        "format": "zcu.migration-run-evidence",
        "format_version": "1.7",
        "entries": [raw],
        "future": {"keep": True},
    }
    evidence_path = request.result_root.parent / "evidence.json"
    evidence_path.write_text(json.dumps(document), encoding="utf-8")
    return replace(request, run_evidence=load_run_evidence(evidence_path)), source


def noop_validation(path: Path, tag: str, cfg_type: str) -> None:
    """Assert an existing native Path with the write_run synthetic identities.

    This callback only checks fixture identity/existence. The owning tests
    separately read the native file to assert its public data contract.
    """
    assert path.is_file()
    assert tag == "synthetic_scan"
    assert cfg_type == "SyntheticCfg"


def manifest_path(request: MigrationRequest) -> Path:
    """Return the request destination manifest Path without filesystem access."""
    return request.results_root / request.name / "records" / "migration-state.json"
