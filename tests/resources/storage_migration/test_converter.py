"""Observable migration, source preservation and explicit resume contracts."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from zcu_tools.datafile import (
    DataVariable,
    JsonObject,
    load_run_data,
)
from zcu_tools.resources.entry import ResultEntry
from zcu_tools.resources.entry.views import FieldView
from zcu_tools.resources.storage_migration import (
    KeyRule,
    MigrationInputError,
    MigrationMapping,
    MigrationRequest,
    ModuleRule,
    load_run_evidence,
    migrate_storage,
)

from tests.resources.storage_migration.fakes import (
    manifest_path,
    noop_validation,
    validate_cfg,
    write_meta,
    write_run,
)

pytestmark = pytest.mark.usefixtures("registry_guard")


@pytest.mark.parametrize("flag, replacement", [(True, 1), (False, 0)])
def test_first_report_boolean_number_collision_is_not_owned(
    request_data: MigrationRequest,
    mapping: MigrationMapping,
    flag: bool,
    replacement: int,
) -> None:
    migrate_storage(
        request_data,
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    state_path = manifest_path(request_data)
    manifest = json.loads(state_path.read_text())
    manifest["report_published"] = False
    manifest["report"]["future_nested"] = {"flags": [flag]}
    state_path.write_text(json.dumps(manifest))
    existing = json.loads(json.dumps(manifest["report"]))
    existing["future_nested"]["flags"][0] = replacement
    report_path = Path(manifest["report_path"])
    report_path.write_text(json.dumps(existing))
    before = report_path.read_bytes()
    with pytest.raises(MigrationInputError, match="unowned report collision"):
        migrate_storage(
            replace(request_data, resume=True),
            mapping=mapping,
            validate_cfg=validate_cfg,
            validate_native=noop_validation,
        )
    assert report_path.read_bytes() == before
    assert (
        json.loads(state_path.read_text())["report"]["future_nested"]["flags"][0]
        is flag
    )


@pytest.mark.parametrize("which", ["result", "database"])
def test_existing_destination_is_never_merged(
    request_data: MigrationRequest, mapping: MigrationMapping, which: str
) -> None:
    target = (
        request_data.results_root if which == "result" else request_data.database_root
    ) / request_data.name
    target.mkdir(parents=True)
    marker = target / "caller.txt"
    marker.write_text("unchanged")
    with pytest.raises(FileExistsError):
        migrate_storage(
            request_data,
            mapping=mapping,
            validate_cfg=validate_cfg,
            validate_native=noop_validation,
        )
    assert marker.read_text() == "unchanged"
    other = (
        request_data.database_root if which == "result" else request_data.results_root
    ) / request_data.name
    assert not other.exists()


def test_resume_refuses_arbitrary_entry(
    request_data: MigrationRequest, mapping: MigrationMapping
) -> None:
    ResultEntry.create(
        request_data.name,
        result_root=request_data.results_root,
        database_root=request_data.database_root,
    )
    with pytest.raises(MigrationInputError, match="manifest"):
        migrate_storage(
            replace(request_data, resume=True),
            mapping=mapping,
            validate_cfg=validate_cfg,
            validate_native=noop_validation,
        )


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
    provenance = a.meta("C1.frequency")
    assert provenance is not None and provenance.stderr == 0.25
    unknown = a.general.ext["unknown"]
    assert isinstance(unknown, FieldView)
    assert unknown["legacy"] == 1
    assert a.general.ext["expression"] == "${md.old_frequency}"
    assert {item.location for item in report.pending} == {"unknown", "expression"}
    assert (source_a.read_bytes(), source_b.read_bytes()) == before
    entry.setup.C1.frequency = 77.0
    assert entry.use_point("context_a").C1.frequency == 12.5


@pytest.mark.parametrize("first", ["parameters", "data"])
def test_parts_accumulate_with_same_identity(
    request_data: MigrationRequest, mapping: MigrationMapping, first: str
) -> None:
    write_meta(request_data, "context", {"old_frequency": 8.0})
    params = request_data.result_root / "chip" / "qubit" / "params.json"
    params.write_text('{"legacy": true}')
    initial = migrate_storage(
        replace(request_data, part=first),
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    second = "data" if first == "parameters" else "parameters"
    result = migrate_storage(
        replace(request_data, part=second, resume=True),
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    assert result.entry_id == initial.entry_id
    assert result.part == "all"
    assert len(result.key_mappings) == 1
    assert len(result.preserved_files) == 1
    assert (
        request_data.results_root / request_data.name / "params.json"
    ).read_bytes() == params.read_bytes()


def test_data_resume_keeps_valid_user_edit(
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
    migrate_storage(
        replace(request_data, part="data", resume=True),
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
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


@pytest.mark.parametrize("change", ["mapping", "source", "report"])
def test_resume_rejects_changed_identity(
    request_data: MigrationRequest, mapping: MigrationMapping, change: str
) -> None:
    migrate_storage(
        replace(request_data, part="parameters"),
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    request = replace(request_data, resume=True)
    if change == "mapping":
        mapping = replace(mapping, mapping_version="1.1")
    elif change == "source":
        (request.result_root / "other" / "qubit").mkdir(parents=True)
        (request.database_root / "other" / "qubit").mkdir(parents=True)
        request = replace(request, source_chip="other")
    else:
        request = replace(
            request, report_path=request.result_root.parent / "another-report.json"
        )
    with pytest.raises(MigrationInputError):
        migrate_storage(
            request,
            mapping=mapping,
            validate_cfg=validate_cfg,
            validate_native=noop_validation,
        )


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
        assert stored.metadata.labber_path is None
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
    request, source = write_run(request_data)
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


def test_report_rebuild_retains_future_fields(
    request_data: MigrationRequest, mapping: MigrationMapping
) -> None:
    migrate_storage(
        replace(request_data, part="parameters"),
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    state_path = manifest_path(request_data)
    manifest = json.loads(state_path.read_text())
    manifest["format_version"] = "1.8"
    manifest["future"] = {"nested": [True, 9]}
    manifest["identity"]["future_identity"] = 7
    manifest["report"]["format_version"] = "1.9"
    manifest["report"]["future_report"] = {"kept": True}
    state_path.write_text(json.dumps(manifest))
    report_path = Path(manifest["report_path"])
    report_path.unlink()
    migrate_storage(
        replace(request_data, part="data", resume=True),
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    after = json.loads(state_path.read_text())
    report = json.loads(report_path.read_text())
    assert after["future"] == manifest["future"]
    assert after["identity"]["future_identity"] == 7
    assert report["future_report"] == {"kept": True}
    assert report["part"] == "all"


def test_preserved_archives_keep_paths_bytes_and_sources(
    request_data: MigrationRequest, mapping: MigrationMapping
) -> None:
    source = request_data.result_root / "chip" / "qubit"
    originals = {
        "arb_waveforms/wave.npz": b"arbitrary wave bytes",
        "context/image/figure.png": b"unknown point image",
        "autofluxdep_runs/run/blob.h5": b"workflow archive",
        "samples.csv": b"old,columns\n1,2\n",
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
        request_data.database_root / request_data.name / "arb_waveforms" / "wave.npz"
    ).read_bytes() == originals["arb_waveforms/wave.npz"]
    preserved = (
        request_data.database_root
        / request_data.name
        / "migration-preserved"
        / "result"
    )
    assert (preserved / "context" / "image" / "figure.png").read_bytes() == originals[
        "context/image/figure.png"
    ]
    assert any(
        item.source == source / "unrecognized.bin"
        and item.location == "unconverted file"
        for item in report.pending
    )
    assert all(
        (source / relative).read_bytes() == content
        for relative, content in originals.items()
    )


def test_owned_copy_hash_conflict_is_not_repaired(
    request_data: MigrationRequest, mapping: MigrationMapping
) -> None:
    source = request_data.result_root / "chip" / "qubit" / "params.json"
    source.write_bytes(b'{"legacy": true}')
    report = migrate_storage(
        replace(request_data, part="parameters"),
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    destination = report.preserved_files[0].destination
    destination.write_bytes(b"caller edit")
    with pytest.raises(MigrationInputError, match="hash"):
        migrate_storage(
            replace(request_data, resume=True, part="data"),
            mapping=mapping,
            validate_cfg=validate_cfg,
            validate_native=noop_validation,
        )
    assert destination.read_bytes() == b"caller edit"
    assert source.read_bytes() == b'{"legacy": true}'


def test_existing_independent_report_is_never_overwritten(
    request_data: MigrationRequest, mapping: MigrationMapping
) -> None:
    path = request_data.result_root.parent / "caller-report.json"
    path.write_bytes(b"caller file")
    with pytest.raises(FileExistsError):
        migrate_storage(
            replace(request_data, report_path=path),
            mapping=mapping,
            validate_cfg=validate_cfg,
            validate_native=noop_validation,
        )
    assert path.read_bytes() == b"caller file"
    assert not (request_data.results_root / request_data.name).exists()


@pytest.mark.parametrize("same_content", [True, False])
def test_first_report_publication_window_requires_same_report(
    request_data: MigrationRequest,
    mapping: MigrationMapping,
    same_content: bool,
) -> None:
    migrate_storage(
        replace(request_data, part="parameters"),
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    state_path = manifest_path(request_data)
    manifest = json.loads(state_path.read_text())
    manifest["report_published"] = False
    state_path.write_text(json.dumps(manifest))
    report_path = Path(manifest["report_path"])
    if same_content:
        result = migrate_storage(
            replace(request_data, part="data", resume=True),
            mapping=mapping,
            validate_cfg=validate_cfg,
            validate_native=noop_validation,
        )
        assert result.part == "all"
    else:
        report_path.write_text('{"unrelated": true}')
        with pytest.raises(MigrationInputError, match="unowned report"):
            migrate_storage(
                replace(request_data, part="data", resume=True),
                mapping=mapping,
                validate_cfg=validate_cfg,
                validate_native=noop_validation,
            )
        assert report_path.read_text() == '{"unrelated": true}'


@pytest.mark.parametrize("phase", ["planned", "prepared", "published"])
def test_native_resume_before_labber_move_keeps_assignment(
    request_data: MigrationRequest,
    mapping: MigrationMapping,
    phase: str,
) -> None:
    request, source = write_run(request_data)
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
    request, source = write_run(request_data)
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
    assert point.meta("C1.channel.ch") is not None


def test_module_candidates_keep_destinations_and_declared_reference_priority(
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
    module = point.C1.module
    assert isinstance(module, FieldView)
    assert module.chosen == "C1.pulses.first_choice"
    destinations = {
        item.old_key: item.new_path
        for item in report.key_mappings
        if item.old_file == library
    }
    assert destinations == {
        "first_choice": "C1.pulses.first_choice",
        "second_choice": "C1.pulses.second_choice",
        "third_choice": "C1.pulses.third_choice",
        "unknown_module": None,
    }
    pending = {
        item.location: item.reason for item in report.pending if item.source == library
    }
    assert set(pending) == {
        "modules.first_choice",
        "modules.second_choice",
        "modules.third_choice",
        "modules.unknown_module",
        "waveforms.pulse",
        "future_root",
    }
    assert all("由 4a 轉換" in reason for reason in pending.values())
    assert "Unknown" in pending["modules.unknown_module"]
