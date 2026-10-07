"""Offline declaration/converter contracts, not individual experiment behavior."""

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from pydantic import TypeAdapter
from zcu_tools.datafile import (
    CfgSnapshot,
    ExperimentPayload,
    JsonObject,
    LabberMetadata,
    LabberPayload,
    load_run_data,
    write_labber,
)
from zcu_tools.resources.storage_migration import (
    LegacyRunEvidence,
    LegacySnapshotEvidence,
    MigrationMapping,
    MigrationRequest,
    load_run_evidence,
    migrate_storage,
)

from tests._native_support import native_metadata
from zcu_lab.migration_experiments import MIGRATION_EXPERIMENTS, MigrationExperiment


@pytest.fixture
def declaration() -> MigrationExperiment:
    """Select the shipped historical pair, not cfg_type alone."""
    return next(
        item
        for item in MIGRATION_EXPERIMENTS
        if (item.source_tag, item.cfg_type)
        == ("twotone/ge/ro_optimize/freq", "FreqCfg")
    )


@pytest.fixture
def historical_cfg() -> CfgSnapshot:
    """Provide unnormalized JSON with derivable readout fields and unknown keys."""
    values: JsonObject = {
        "modules": {
            "readout": {
                "type": "readout/pulse",
                "pulse_cfg": {
                    "waveform": {"style": "const", "length": 1.0},
                    "ch": 1,
                    "nqz": 2,
                    "freq": 6100.0,
                    "gain": 0.2,
                },
                "ro_cfg": {"ro_ch": 2, "ro_length": 1.0, "trig_offset": 0.5},
            },
            "qub_pulse": {
                "type": "pulse",
                "waveform": {"style": "const", "length": 0.1},
                "ch": 3,
                "nqz": 2,
                "freq": 4000.0,
                "gain": 0.1,
            },
        },
        "sweep": {
            "freq": {"start": 4000.0, "stop": 5000.0, "expts": 2, "step": 1000.0}
        },
        "future_cfg": {"nested": [{"keep": True}]},
    }
    return CfgSnapshot(values=values, cfg_type="FreqCfg", schema_version="1.0")


def test_declaration_validation_preserves_unnormalized_nested_cfg(
    historical_cfg: CfgSnapshot,
    declaration: MigrationExperiment,
) -> None:
    before = deepcopy(historical_cfg)
    declaration.validate_cfg(historical_cfg)
    assert historical_cfg == before


@pytest.mark.parametrize("version", ["1.bad", "1", "1.0.1"])
def test_declaration_rejects_malformed_complete_cfg_version(
    historical_cfg: CfgSnapshot,
    declaration: MigrationExperiment,
    version: str,
) -> None:
    cfg = replace(historical_cfg, schema_version=version)
    before = deepcopy(cfg)
    with pytest.raises(ValueError, match=r"/cfg.*major.minor"):
        declaration.validate_cfg(cfg)
    assert cfg == before


@pytest.fixture
def migration_request(
    tmp_path: Path,
    historical_cfg: CfgSnapshot,
    declaration: MigrationExperiment,
) -> tuple[MigrationRequest, MigrationMapping, Path]:
    """Build one raw historical run and evidence without runtime registration."""
    request = MigrationRequest(
        result_root=tmp_path / "result",
        database_root=tmp_path / "Database",
        results_root=tmp_path / "results",
        source_chip="chip",
        source_qubit="qubit",
        name="new",
        part="data",
    )
    (request.result_root / "chip/qubit").mkdir(parents=True)
    source = request.database_root / "chip/qubit/scan.hdf5"
    source.parent.mkdir(parents=True)
    schema = declaration.schemas[0]
    write_labber(
        source,
        ExperimentPayload(
            variables={
                schema.variable: LabberPayload(
                    (
                        schema.signal_name,
                        schema.signal_unit,
                        np.array([1, 2], dtype=schema.signal_dtype),
                    ),
                    [(schema.axes[0].name, schema.axes[0].unit, np.array([4e9, 5e9]))],
                )
            },
            metadata=LabberMetadata(tags=[declaration.source_tag]),
            representation="single",
        ),
        cfg=historical_cfg,
    )
    evidence = LegacyRunEvidence(
        source=Path("scan.hdf5"),
        source_hash=hashlib.sha256(source.read_bytes()).hexdigest(),
        experiment=declaration.source_tag,
        cfg=historical_cfg,
        started_at="2025-10-01T00:00:00Z",
        finished_at=None,
        completion="stopped",
        snapshot=LegacySnapshotEvidence(
            entry_name="historical", point=None, description=None, roles={}, params={}
        ),
        provenance=native_metadata().provenance,
    )
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "format": "zcu.migration-run-evidence",
                "format_version": "1.7",
                "entries": [
                    TypeAdapter(LegacyRunEvidence).dump_python(evidence, mode="json")
                ],
                "future": {"nested": [{"keep": True}]},
            }
        )
    )
    mapping = MigrationMapping(
        mapping_version="1.0",
        components={},
        rules=(),
        roles={},
        data_schemas={
            (declaration.source_tag, declaration.cfg_type): declaration.schemas
        },
    )
    return (
        replace(request, run_evidence=load_run_evidence(evidence_path)),
        mapping,
        source,
    )


@pytest.mark.parametrize("version", ["1.bad", "1", "1.0.1", "2.0"])
def test_unconvertible_cfg_version_is_pending_and_can_be_corrected(
    migration_request: tuple[MigrationRequest, MigrationMapping, Path],
    declaration: MigrationExperiment,
    version: str,
) -> None:
    request, mapping, source = migration_request
    assert request.run_evidence is not None
    path = request.result_root.parent / "evidence.json"
    document = json.loads(path.read_text())
    document["entries"][0]["cfg"]["schema_version"] = version
    path.write_text(json.dumps(document))
    request = replace(request, run_evidence=load_run_evidence(path))
    before = source.read_bytes()

    def check_cfg(tag: str, cfg: CfgSnapshot) -> None:
        assert tag == declaration.source_tag
        declaration.validate_cfg(cfg)

    def check_native(path: Path, tag: str, cfg_type: str) -> None:
        assert source.read_bytes() == before
        declaration.validate_native(path)

    report = migrate_storage(
        request, mapping=mapping, validate_cfg=check_cfg, validate_native=check_native
    )
    assert report.converted_files == report.moved_labber_files == ()
    assert any(
        item.source == source and "cfg_schema_version" in item.reason
        for item in report.pending
    )
    assert source.read_bytes() == before
    manifest_path = request.results_root / request.name / "records/migration-state.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["runs"] == []
    assert not any(item["operation"] == "native" for item in manifest["files"])
    assert manifest["evidence"] == document

    document["entries"][0]["cfg"]["schema_version"] = "1.7"
    path.write_text(json.dumps(document))
    corrected = replace(request, resume=True, run_evidence=load_run_evidence(path))
    result = migrate_storage(
        corrected, mapping=mapping, validate_cfg=check_cfg, validate_native=check_native
    )
    assert result.entry_id == report.entry_id
    assert len(result.converted_files) == len(result.moved_labber_files) == 1
    stored = load_run_data(result.converted_files[0].destination)
    assert stored.cfg.values == document["entries"][0]["cfg"]["values"]
    assert stored.cfg.schema_version == "1.7"
    assert result.moved_labber_files[0].destination.read_bytes() == before


@pytest.mark.parametrize("dry_run", [True, False])
def test_converter_preserves_raw_cfg_during_validation(
    migration_request: tuple[MigrationRequest, MigrationMapping, Path],
    declaration: MigrationExperiment,
    dry_run: bool,
) -> None:
    request, mapping, source = migration_request
    before = deepcopy(request.run_evidence)
    source_bytes = source.read_bytes()

    def check_cfg(tag: str, cfg: CfgSnapshot) -> None:
        declaration.validate_cfg(cfg)

    def check_native(path: Path, tag: str, cfg_type: str) -> None:
        assert source.read_bytes() == source_bytes
        declaration.validate_native(path)

    report = migrate_storage(
        replace(request, dry_run=dry_run),
        mapping=mapping,
        validate_cfg=check_cfg,
        validate_native=check_native,
    )
    assert request.run_evidence == before
    if dry_run:
        assert source.read_bytes() == source_bytes
        assert not (request.database_root / request.name).exists()
        assert not (request.results_root / request.name).exists()
    else:
        assert before is not None and before.entries[0].cfg is not None
        assert len(report.converted_files) == 1
        stored = load_run_data(report.converted_files[0].destination)
        assert stored.cfg == before.entries[0].cfg
        assert report.moved_labber_files[0].destination.read_bytes() == source_bytes
