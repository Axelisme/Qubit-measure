"""Legacy exps points through migrate_storage and ResultEntry's public contracts."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import TypeAdapter
from zcu_tools.resources.entry import ResultEntry

from migrations.storage_migration import (
    MigrationInputError,
    MigrationManifest,
    MigrationMapping,
    MigrationReport,
    MigrationRequest,
    ModuleRule,
    migrate_storage,
)
from tests.migrations.storage_migration.fakes import (
    manifest_path,
    noop_validation,
    validate_cfg,
    write_meta,
)

pytestmark = pytest.mark.usefixtures("registry_guard")


def _convert(request: MigrationRequest, mapping: MigrationMapping) -> MigrationReport:
    return migrate_storage(
        request,
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )


def _entry(request: MigrationRequest) -> ResultEntry:
    return ResultEntry.open(
        request.name,
        result_root=request.results_root,
        database_root=request.database_root,
    )


def _backup_result(request: MigrationRequest) -> Path:
    return request.result_root.parent.parent / "repo-backup/result/chip/qubit"


def _image(request: MigrationRequest, label: str, contents: bytes = b"figure") -> Path:
    path = request.result_root / "chip" / "qubit" / "exps" / label / "image" / "fit.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    return path


@pytest.fixture
def exps_mapping(mapping: MigrationMapping) -> MigrationMapping:
    return replace(
        mapping,
        module_rules=(
            ModuleRule(
                old_name="constant",
                target_path="C1.waveforms.constant",
                reference_path=None,
                reason="Explicit waveform mapping",
                kind="waveform",
            ),
        ),
    )


@pytest.mark.parametrize("metadata", ["values", "empty", "absent"])
def test_exps_child_becomes_complete_point_with_independent_meta_and_library(
    request_data: MigrationRequest,
    exps_mapping: MigrationMapping,
    metadata: str,
) -> None:
    request = replace(request_data, part="parameters")
    write_meta(request, "", {"old_frequency": 3.0})
    write_meta(request, "direct", {"old_frequency": 4.0})
    child = request.result_root / "chip/qubit/exps/070716_0.500"
    child.mkdir(parents=True)
    if metadata != "absent":
        write_meta(
            request,
            "exps/070716_0.500",
            {"old_frequency": 12.5, "unknown": "=md.frequency"}
            if metadata == "values"
            else {},
        )
    library = child / "module_cfg.yaml"
    library.write_text(
        json.dumps(
            {
                "modules": {},
                "waveforms": {
                    "constant": {"style": "const", "length": 0.125},
                    "expression": {"style": "const", "length": "=md.length"},
                },
            }
        ),
        encoding="utf-8",
    )
    original = {path: path.read_bytes() for path in child.iterdir()}
    report = _convert(request, exps_mapping)
    entry = _entry(request)
    assert entry.list_points() == ["070716_0.500", "direct"]
    point = entry.use_point("070716_0.500")
    assert point.C1.frequency == (12.5 if metadata == "values" else 3.0)
    assert entry.setup.C1.frequency == 3.0
    assert entry.use_point("direct").C1.frequency == 4.0
    snapshot = entry.module_library("070716_0.500").snapshot()
    assert snapshot.waveform_keys() == ("C1.waveforms.constant",)
    assert snapshot.get_waveform("C1.waveforms.constant").length == 0.125
    assert any(
        item.source == library and item.location == "waveforms.expression"
        for item in report.pending
    )
    if metadata == "values":
        assert point.general.ext["unknown"] == "=md.frequency"
        assert any(item.location == "unknown" for item in report.pending)
    assert {path: path.read_bytes() for path in child.iterdir()} == original
    assert report.failures == ()


def test_entry_figures_keep_relative_paths_and_workflow_images_stay_archived(
    request_data: MigrationRequest, mapping: MigrationMapping
) -> None:
    source = request_data.result_root / "chip/qubit"
    files = {
        "image/fit.png": b"root figure",
        "overview/image/fit.png": b"nested figure",
        "autofluxdep_runs/run/image/fit.png": b"workflow figure",
    }
    for relative, content in files.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    dry = _convert(replace(request_data, dry_run=True), mapping)
    assert not (request_data.results_root / request_data.name).exists()
    report = _convert(request_data, mapping)
    entry = _entry(request_data)
    assert (entry.result_path / "figures/image/fit.png").is_file()
    assert (
        entry.result_path / "figures/overview/image/fit.png"
    ).read_bytes() == b"nested figure"
    workflow = (
        entry.database_path
        / "migration-preserved/result/autofluxdep_runs/run/image/fit.png"
    )
    assert workflow.read_bytes() == b"workflow figure"
    assert dry.failures == ()
    assert not any(
        item.location == "association" and item.source == source / "image/fit.png"
        for item in report.pending
    )
    _convert(replace(request_data, resume=True), mapping)
    assert (entry.result_path / "figures/image/fit.png").read_bytes() == b"root figure"


def test_exps_child_without_parameter_files_still_gets_point(
    request_data: MigrationRequest, mapping: MigrationMapping
) -> None:
    child = request_data.result_root / "chip/qubit/exps/empty"
    child.mkdir(parents=True)
    _convert(replace(request_data, part="parameters"), mapping)
    entry = _entry(request_data)
    assert entry.list_points() == ["empty"]
    assert entry.use_point("empty").C1.frequency == 1.0
    assert list(child.iterdir()) == []


def test_data_publishes_point_figures_once_without_archiving_or_unproven_association(
    request_data: MigrationRequest, mapping: MigrationMapping
) -> None:
    request = replace(request_data, part="parameters")
    write_meta(request, "exps/070716_0.500", {"old_frequency": 12.5})
    figure = _image(request, "070716_0.500", b"point figure bytes")
    outside = request.result_root / "chip/qubit/image/root.png"
    outside.parent.mkdir()
    outside.write_bytes(b"unassociated root image")
    _convert(request, mapping)
    entry = _entry(request)
    target = entry.result_path / "points/070716_0.500/figures/fit.png"
    assert not target.exists(), "parameters-only must not publish data assets"
    report = _convert(replace(request, part="data", resume=True), mapping)
    assert target.is_file(), "exps images belong to their point figures directory"
    backup = _backup_result(request)
    retained_figure = backup / "exps/070716_0.500/image/fit.png"
    assert target.read_bytes() == retained_figure.read_bytes() == b"point figure bytes"
    copies = [item for item in report.preserved_files if item.source == figure]
    assert len(copies) == 1
    assert copies[0].destination == target
    assert copies[0].source_hash == copies[0].destination_hash
    assert copies[0].source_hash == hashlib.sha256(b"point figure bytes").hexdigest()
    assert not any(item.source == figure for item in report.pending)
    archive = entry.database_path / "migration-preserved/result"
    assert not (archive / "exps/070716_0.500/image/fit.png").exists()
    retained_outside = backup / "image/root.png"
    assert (
        entry.result_path / "figures/image/root.png"
    ).read_bytes() == retained_outside.read_bytes()
    assert retained_outside.read_bytes() == b"unassociated root image"
    assert not any(
        item.source == retained_outside and item.location == "association"
        for item in report.pending
    )
    manifest = TypeAdapter(MigrationManifest).validate_json(
        manifest_path(request).read_bytes()
    )
    states = [item for item in manifest.files if item.source == figure]
    assert len(states) == 1
    assert states[0].operation == "copy"
    assert states[0].destination == target
    entry.use_point("070716_0.500").C1.frequency = 99.0
    resumed = _convert(replace(request, part="all", resume=True), mapping)
    assert resumed.preserved_files == report.preserved_files
    assert _entry(request).use_point("070716_0.500").C1.frequency == 99.0
    assert target.read_bytes() == retained_figure.read_bytes()
    assert not (request.result_root / "chip/qubit").exists()
    assert not (request.database_root / "chip/qubit").exists()


@pytest.mark.parametrize("changed", ["backup", "destination"])
def test_resume_refuses_changed_point_figure_hash(
    request_data: MigrationRequest, mapping: MigrationMapping, changed: str
) -> None:
    write_meta(request_data, "exps/context", {})
    figure = _image(request_data, "context")
    _convert(request_data, mapping)
    target = _entry(request_data).result_path / "points/context/figures/fit.png"
    assert target.is_file(), "point figure must be published before resume"
    path = (
        _backup_result(request_data)
        / figure.relative_to(request_data.result_root / "chip/qubit")
        if changed == "backup"
        else target
    )
    path.write_bytes(b"changed")
    with pytest.raises(
        MigrationInputError, match="backup" if changed == "backup" else "hash"
    ):
        _convert(replace(request_data, resume=True), mapping)
    assert path.read_bytes() == b"changed"


def test_data_first_defers_figures_until_parameters_create_the_point(
    request_data: MigrationRequest, mapping: MigrationMapping
) -> None:
    write_meta(request_data, "exps/070716_0.500", {"old_frequency": 12.5})
    figure = _image(request_data, "070716_0.500")
    request = replace(request_data, part="data")
    report = _convert(request, mapping)
    entry = _entry(request)
    assert not (entry.result_path / "points/070716_0.500").exists()
    assert not any(item.source == figure for item in report.preserved_files)
    pending = [item for item in report.pending if item.source == figure]
    assert len(pending) == 1
    assert "point" in pending[0].reason.lower()
    _convert(replace(request, part="parameters", resume=True), mapping)
    assert _entry(request).list_points() == ["070716_0.500"]
    completed = _convert(replace(request, resume=True), mapping)
    target = entry.result_path / "points/070716_0.500/figures/fit.png"
    retained = _backup_result(request) / "exps/070716_0.500/image/fit.png"
    assert target.read_bytes() == retained.read_bytes() == b"figure"
    digest = hashlib.sha256(b"figure").hexdigest()
    publication = next(
        item for item in completed.preserved_files if item.source == figure
    )
    assert publication.source_hash == publication.destination_hash == digest
    assert not (request.result_root / "chip/qubit").exists()
    assert not (request.database_root / "chip/qubit").exists()
    assert not any(item.source == figure for item in completed.pending)
    assert (
        len([item for item in completed.preserved_files if item.source == figure]) == 1
    )
    resumed = _convert(replace(request, resume=True), mapping)
    assert resumed.preserved_files == completed.preserved_files
    assert target.read_bytes() == retained.read_bytes() == b"figure"


@pytest.mark.parametrize("dry_run", [False, True])
def test_duplicate_direct_and_exps_labels_fail_with_both_sources_before_point_writes(
    request_data: MigrationRequest, mapping: MigrationMapping, dry_run: bool
) -> None:
    direct = write_meta(request_data, "same", {"old_frequency": 7.0}).parent
    nested = write_meta(request_data, "exps/same", {"old_frequency": 8.0}).parent
    with pytest.raises(MigrationInputError) as caught:
        _convert(replace(request_data, part="parameters", dry_run=dry_run), mapping)
    assert str(direct) in str(caught.value)
    assert str(nested) in str(caught.value)
    if not dry_run:
        assert _entry(request_data).list_points() == []
    assert json.loads((direct / "meta_info.json").read_text()) == {"old_frequency": 7.0}
    assert json.loads((nested / "meta_info.json").read_text()) == {"old_frequency": 8.0}


def test_dry_run_maps_exps_without_creating_destinations_or_changing_sources(
    request_data: MigrationRequest, mapping: MigrationMapping
) -> None:
    meta = write_meta(request_data, "exps/070716_0.500", {"old_frequency": 12.5})
    figure = _image(request_data, "070716_0.500")
    original = (meta.read_bytes(), figure.read_bytes())
    report = _convert(replace(request_data, dry_run=True), mapping)
    rows = [item for item in report.key_mappings if item.old_file == meta]
    assert len(rows) == 1
    assert rows[0].new_path == "C1.frequency"
    assert rows[0].new_file == (
        request_data.results_root / request_data.name / "points/070716_0.500/point.yaml"
    )
    assert not any(item.source == figure for item in report.pending)
    assert not request_data.results_root.exists()
    assert not (request_data.database_root / request_data.name).exists()
    assert (meta.read_bytes(), figure.read_bytes()) == original
