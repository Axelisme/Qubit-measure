"""Native and preserved data publication, validation and interrupted source moves."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import h5py as h5
import numpy as np
import pytest
from zcu_tools.datafile import DataVariable, load_run_data

from migrations.storage_migration import (
    MigrationInputError,
    MigrationMapping,
    MigrationRequest,
    load_run_evidence,
    migrate_storage,
)
from tests._native_support import hdf_group
from tests.migrations.storage_migration.fakes import (
    manifest_path,
    noop_validation,
    validate_cfg,
    write_meta,
    write_run,
)

pytestmark = pytest.mark.usefixtures("registry_guard")


@pytest.mark.parametrize("unsupported", [False, True])
def test_instrument_snapshot_preserves_values_or_reports_isolated_pending(
    request_data: MigrationRequest, mapping: MigrationMapping, unsupported: bool
) -> None:
    request_data, source = write_run(request_data)
    with h5.File(source, "r+") as file:
        instrument = hdf_group(file, "Instrument config").create_group(
            "synthetic source"
        )
        instrument.attrs["frequency"] = np.float64(5.25e9)
        instrument.attrs["enabled"] = np.bool_(True)
        instrument.attrs["label"] = np.bytes_(b"offline")
        instrument.attrs["points"] = np.array([1 + 2j, complex(np.inf, np.nan)])
        instrument.attrs["bounds"] = np.array([np.nan, np.inf, -np.inf])
        instrument.attrs["empty"] = np.array([], dtype=float)
        if unsupported:
            instrument.attrs["reference"] = file.ref
    evidence_path = request_data.result_root.parent / "evidence.json"
    raw = json.loads(evidence_path.read_text())
    raw["entries"][0]["source_hash"] = hashlib.sha256(source.read_bytes()).hexdigest()
    evidence_path.write_text(json.dumps(raw))
    request_data = replace(request_data, run_evidence=load_run_evidence(evidence_path))
    original = source.read_bytes()
    report = migrate_storage(
        request_data,
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    native = next(
        (request_data.database_root / request_data.name / "runs").glob("*/data.h5")
    )
    stored = load_run_data(native)
    if unsupported:
        assert stored.metadata.device_snapshot is None
        pending = [
            item
            for item in report.pending
            if item.location == "device snapshot pending"
        ]
        assert len(pending) == 1
        assert "reference" in pending[0].reason
    else:
        assert stored.metadata.device_snapshot is not None
        assert stored.metadata.device_snapshot["synthetic source"] == {
            "frequency": 5.25e9,
            "enabled": True,
            "label": "offline",
            "points": [
                {"real": 1.0, "imag": 2.0},
                {"real": {"nonfinite": "inf"}, "imag": {"nonfinite": "nan"}},
            ],
            "bounds": [
                {"nonfinite": "nan"},
                {"nonfinite": "inf"},
                {"nonfinite": "-inf"},
            ],
            "empty": [],
        }
        assert not report.pending
    np.testing.assert_array_equal(
        stored.payload.variables[DataVariable("signal")].z, [1 + 2j, 3 + 4j]
    )
    assert (
        request_data.database_root / request_data.name / "Labber/2025/10/scan.hdf5"
    ).read_bytes() == original
    assert not report.failures


def test_native_links_to_preserved_labber_relative_to_entry_database(
    request_data: MigrationRequest, mapping: MigrationMapping
) -> None:
    request_data, source = write_run(request_data)
    original = source.read_bytes()
    report = migrate_storage(
        request_data,
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    native = next(
        (request_data.database_root / request_data.name / "runs").glob("*/data.h5")
    )
    stored = load_run_data(native)
    assert stored.metadata.labber_path == "Labber/2025/10/scan.hdf5"
    assert (
        request_data.database_root / request_data.name / stored.metadata.labber_path
    ).read_bytes() == original
    assert not report.failures


def test_native_is_readable_and_validated_before_source_removal(
    request_data: MigrationRequest, mapping: MigrationMapping
) -> None:
    request, source = write_run(request_data)
    before = source.read_bytes()
    observed: list[Path] = []

    def validate(path: Path, tag: str, cfg_type: str) -> None:
        assert source.read_bytes() == before
        stored = load_run_data(path)
        assert stored.metadata.experiment == tag
        assert stored.cfg.cfg_type == cfg_type
        assert stored.metadata.completion == "stopped"
        assert stored.metadata.snapshot.point == "point_at_acquisition"
        assert stored.metadata.snapshot.entry_name == "historical"
        assert stored.metadata.labber_path == "Labber/2025/10/scan.hdf5"
        assert stored.cfg.values == {"frequency": 12.5}
        np.testing.assert_array_equal(
            stored.payload.variables[DataVariable("signal")].z, [1 + 2j, 3 + 4j]
        )
        observed.append(path)

    report = migrate_storage(
        request, mapping=mapping, validate_cfg=validate_cfg, validate_native=validate
    )
    assert len(observed) == 1
    assert len(report.converted_files) == len(report.moved_labber_files) == 1
    assert not source.exists()
    moved = (
        request.database_root / request.name / "Labber" / "2025" / "10" / "scan.hdf5"
    )
    assert moved.read_bytes() == before
    stored = load_run_data(observed[0])
    assert stored.metadata.snapshot.entry_id == report.entry_id
    manifest = json.loads(manifest_path(request).read_text())
    assert request.run_evidence is not None
    assert manifest["evidence"] == request.run_evidence.raw


def test_validation_failure_retains_source_and_reuses_run_identity(
    request_data: MigrationRequest, mapping: MigrationMapping
) -> None:
    request, source = write_run(request_data)
    paths: list[Path] = []

    def fail(path: Path, tag: str, cfg_type: str) -> None:
        noop_validation(path, tag, cfg_type)
        paths.append(path)
        raise RuntimeError("typed spec rejected")

    with pytest.raises(RuntimeError, match="typed spec rejected"):
        migrate_storage(
            request, mapping=mapping, validate_cfg=validate_cfg, validate_native=fail
        )
    assert source.is_file()
    before = json.loads(manifest_path(request).read_text())
    assert before["files"][0]["phase"] == "published"
    assert not before["files"][0]["native_validated"]

    def succeed(path: Path, tag: str, cfg_type: str) -> None:
        assert source.is_file()
        assert path == paths[0]
        noop_validation(path, tag, cfg_type)

    report = migrate_storage(
        replace(request, resume=True, run_evidence=None),
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=succeed,
    )
    after = json.loads(manifest_path(request).read_text())
    assert after["runs"] == before["runs"]
    assert not report.failures
    assert not source.exists()


def test_validated_native_hash_conflict_retains_labber_source(
    request_data: MigrationRequest, mapping: MigrationMapping
) -> None:
    request, source = write_run(request_data)

    def fail(path: Path, tag: str, cfg_type: str) -> None:
        noop_validation(path, tag, cfg_type)
        raise RuntimeError("pause")

    with pytest.raises(RuntimeError):
        migrate_storage(
            request, mapping=mapping, validate_cfg=validate_cfg, validate_native=fail
        )
    manifest = json.loads(manifest_path(request).read_text())
    manifest["files"][0]["native_validated"] = True
    native = Path(manifest["files"][0]["destination"])
    native.write_bytes(b"changed")
    manifest_path(request).write_text(json.dumps(manifest))
    with pytest.raises(MigrationInputError, match="hash"):
        migrate_storage(
            replace(request, resume=True),
            mapping=mapping,
            validate_cfg=validate_cfg,
            validate_native=noop_validation,
        )
    assert source.is_file()
    assert native.read_bytes() == b"changed"


def test_missing_evidence_is_pending_and_can_be_supplied_on_resume(
    request_data: MigrationRequest, mapping: MigrationMapping
) -> None:
    request, source = write_run(request_data)
    initial = migrate_storage(
        replace(request, run_evidence=None, part="data"),
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    assert source.is_file()
    assert initial.pending
    assert not initial.moved_labber_files
    report = migrate_storage(
        replace(request, resume=True, part="data"),
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    assert report.entry_id == initial.entry_id
    assert not report.pending
    assert not source.exists()


def test_dry_run_never_publishes_or_calls_validator(
    request_data: MigrationRequest, mapping: MigrationMapping
) -> None:
    request, source = write_run(request_data)
    write_meta(request, "context", {"old_frequency": 8.0})

    def forbidden(path: Path, tag: str, cfg_type: str) -> None:
        pytest.fail(f"dry-run called validator: {path} {tag}")

    result = migrate_storage(
        replace(request, dry_run=True),
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=forbidden,
    )
    assert result.key_mappings
    assert source.is_file()
    assert not request.results_root.exists()
    assert not (request.database_root / request.name).exists()


@pytest.mark.parametrize(
    "phase", ["planned", "prepared", "published", "source_removed"]
)
def test_move_resumes_each_publication_window(
    request_data: MigrationRequest, mapping: MigrationMapping, phase: str
) -> None:
    request, source = write_run(replace(request_data, part="data"))
    report = migrate_storage(
        request,
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    moved = report.moved_labber_files[0]
    state_path = manifest_path(request)
    manifest = json.loads(state_path.read_text())
    state = next(
        item for item in manifest["files"] if item["operation"] == "move_labber"
    )
    state["phase"] = phase
    if phase != "source_removed":
        source.write_bytes(moved.destination.read_bytes())
    if phase in ("planned", "prepared"):
        if phase == "prepared":
            Path(state["temp_path"]).write_bytes(moved.destination.read_bytes())
        else:
            state["destination_hash"] = None
            Path(state["temp_path"]).write_bytes(b"interrupted partial copy")
        moved.destination.unlink()
    manifest["report"]["moved_labber_files"] = []
    state_path.write_text(json.dumps(manifest))
    result = migrate_storage(
        replace(request, resume=True),
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    assert len(result.moved_labber_files) == 1
    assert result.moved_labber_files[0].destination_hash == moved.destination_hash
    assert not source.exists()
    assert not Path(state["temp_path"]).exists()


def test_preserved_files_keep_entry_relative_paths_bytes_and_sources(
    request_data: MigrationRequest, mapping: MigrationMapping
) -> None:
    source = request_data.result_root / "chip" / "qubit"
    originals = {
        "arb_waveforms/wave.npz": b"arbitrary wave bytes",
        "context/image/figure.png": b"unknown point image",
        "autofluxdep_runs/run/blob.h5": b"workflow archive",
        "unrecognized.bin": b"retained only at source",
    }
    for relative, content in originals.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    report = migrate_storage(
        replace(request_data, part="data"),
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    assert len(report.preserved_files) == 4
    for item in report.preserved_files:
        assert item.destination.read_bytes() == item.source.read_bytes()
        assert item.source_hash == item.destination_hash
    assert (
        request_data.results_root / request_data.name / "arb_waveforms" / "wave.npz"
    ).read_bytes() == originals["arb_waveforms/wave.npz"]
    figures = request_data.results_root / request_data.name / "figures"
    assert (figures / "context/image/figure.png").read_bytes() == originals[
        "context/image/figure.png"
    ]
    entry = request_data.results_root / request_data.name
    assert (entry / "unrecognized.bin").read_bytes() == originals["unrecognized.bin"]
    assert (entry / "autofluxdep_runs/run/blob.h5").read_bytes() == originals[
        "autofluxdep_runs/run/blob.h5"
    ]
    assert report.converted_files == ()
    assert all(
        (source / relative).read_bytes() == content
        for relative, content in originals.items()
    )


@pytest.mark.parametrize("phase", ["planned", "prepared", "published"])
def test_native_resume_before_labber_move_keeps_assignment(
    request_data: MigrationRequest,
    mapping: MigrationMapping,
    phase: str,
) -> None:
    request, source = write_run(replace(request_data, part="data"))
    report = migrate_storage(
        request,
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    state_path = manifest_path(request)
    manifest = json.loads(state_path.read_text())
    assignment = manifest["runs"]
    native = next(item for item in manifest["files"] if item["operation"] == "native")
    source.write_bytes(report.moved_labber_files[0].destination.read_bytes())
    report.moved_labber_files[0].destination.unlink()
    manifest["files"] = [native]
    manifest["report"]["moved_labber_files"] = []
    manifest["report"]["converted_files"] = []
    native["phase"] = phase
    native["native_validated"] = False
    destination = Path(native["destination"])
    if phase == "prepared":
        destination.rename(native["temp_path"])
    elif phase == "planned":
        destination.unlink()
        native["destination_hash"] = None
    state_path.write_text(json.dumps(manifest))

    def validate(path: Path, tag: str, cfg_type: str) -> None:
        assert source.is_file()
        assert path == destination
        noop_validation(path, tag, cfg_type)

    result = migrate_storage(
        replace(request, resume=True),
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=validate,
    )
    assert json.loads(state_path.read_text())["runs"] == assignment
    assert len(result.converted_files) == len(result.moved_labber_files) == 1
    assert not source.exists()


def test_prepared_final_window_does_not_require_temp(
    request_data: MigrationRequest, mapping: MigrationMapping
) -> None:
    request, source = write_run(replace(request_data, part="data"))
    report = migrate_storage(
        request,
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    state_path = manifest_path(request)
    manifest = json.loads(state_path.read_text())
    state = next(
        item for item in manifest["files"] if item["operation"] == "move_labber"
    )
    source.write_bytes(report.moved_labber_files[0].destination.read_bytes())
    state["phase"] = "prepared"
    manifest["report"]["moved_labber_files"] = []
    state_path.write_text(json.dumps(manifest))
    result = migrate_storage(
        replace(request, resume=True),
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    assert len(result.moved_labber_files) == 1
    assert not source.exists()


def test_callback_cannot_change_native_and_authorize_source_deletion(
    request_data: MigrationRequest, mapping: MigrationMapping
) -> None:
    request, source = write_run(request_data)

    def change(path: Path, tag: str, cfg_type: str) -> None:
        noop_validation(path, tag, cfg_type)
        path.write_bytes(b"modified by callback")

    with pytest.raises(MigrationInputError, match="hash"):
        migrate_storage(
            request, mapping=mapping, validate_cfg=validate_cfg, validate_native=change
        )
    assert source.is_file()
