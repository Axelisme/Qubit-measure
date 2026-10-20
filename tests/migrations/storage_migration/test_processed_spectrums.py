"""Processed legacy collections convert through the migration and current I/O seams."""

from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
from zcu_tools.analysis.fluxdep.io import load_spectrums

from migrations.storage_migration import (
    MigrationInputError,
    MigrationMapping,
    MigrationReport,
    MigrationRequest,
    migrate_storage,
)

pytestmark = pytest.mark.usefixtures("registry_guard")


def _group(handle: h5py.Group, path: str) -> h5py.Group:
    item = handle[path]
    if not isinstance(item, h5py.Group):
        raise TypeError(f"Expected HDF5 group at {path}")
    return item


def _dataset(handle: h5py.Group, path: str) -> h5py.Dataset:
    item = handle[path]
    if not isinstance(item, h5py.Dataset):
        raise TypeError(f"Expected HDF5 dataset at {path}")
    return item


def _convert(request: MigrationRequest, mapping: MigrationMapping) -> MigrationReport:
    return migrate_storage(
        request,
        mapping=mapping,
        validate_cfg=lambda _tag, _cfg: None,
        validate_native=lambda _path, _tag, _cfg_type: None,
    )


def _legacy_collection(request: MigrationRequest, frequency_major: bool) -> Path:
    source = request.result_root / "chip/qubit/data/fluxdep/spectrums.hdf5"
    source.parent.mkdir(parents=True)
    signals = np.array([[1 + 2j, 3 - 4j, 5j], [6j, 7 + 8j, 9 - 1j]], dtype=np.complex64)
    with h5py.File(source, "w") as handle:
        group = handle.create_group("scan.with.dots")
        group.attrs["type"] = "TwoTone"
        group["mA_c"] = 2.0
        group["period"] = 4.0
        spectrum = group.create_group("spectrum")
        spectrum["mAs"] = np.array([0.0, 4.0])
        spectrum["fpts"] = np.array([4.0, 5.0, 6.0])
        spectrum["data"] = signals.T if frequency_major else signals
        points = group.create_group("points")
        points["mAs"] = np.array([1.0, 3.0])
        points["fpts"] = np.array([4.25, 5.75])
    return source


@pytest.mark.parametrize("frequency_major", [False, True])
def test_processed_spectrums_convert_axes_calibration_and_complex_values(
    request_data: MigrationRequest, mapping: MigrationMapping, frequency_major: bool
) -> None:
    source = _legacy_collection(request_data, frequency_major)
    original = source.read_bytes()
    report = _convert(request_data, mapping)
    target = report.destination.result_path / "data/fluxdep/spectrums.hdf5"
    assert any(
        item.source == source and item.destination == target
        for item in report.converted_files
    ), "Processed spectrums must be converted, not copied as generic raw bytes"
    actual = load_spectrums(str(target))
    assert list(actual) == ["scan.with.dots"]
    spectrum = actual["scan.with.dots"]
    assert spectrum["flux_half"] == 2.0
    assert spectrum["flux_int"] == 4.0
    assert spectrum["flux_period"] == 4.0
    assert spectrum.get("type") == "TwoTone"
    np.testing.assert_array_equal(spectrum["spectrum"]["dev_values"], [0.0, 4.0])
    np.testing.assert_array_equal(spectrum["spectrum"]["freqs"], [4.0, 5.0, 6.0])
    np.testing.assert_array_equal(spectrum["spectrum"]["fluxs"], [0.0, 1.0])
    np.testing.assert_array_equal(
        spectrum["spectrum"]["signals"],
        [[1 + 2j, 3 - 4j, 5j], [6j, 7 + 8j, 9 - 1j]],
    )
    assert spectrum["spectrum"]["signals"].dtype == np.dtype("complex64")
    np.testing.assert_array_equal(spectrum["points"]["dev_values"], [1.0, 3.0])
    np.testing.assert_array_equal(spectrum["points"]["freqs"], [4.25, 5.75])
    np.testing.assert_array_equal(spectrum["points"]["fluxs"], [0.25, 0.75])
    assert not any(item.destination == target for item in report.preserved_files)
    assert not any(item.source == source for item in report.pending)
    assert report.backup_path is not None
    assert (
        report.backup_path / "result/chip/qubit/data/fluxdep/spectrums.hdf5"
    ).read_bytes() == original
    assert not source.exists()
    before = target.read_bytes()
    resumed = _convert(replace(request_data, resume=True), mapping)
    assert resumed.converted_files == report.converted_files
    assert target.read_bytes() == before


@pytest.mark.parametrize("relative", ["spectrums.hdf5", "spectrums.legacy.hdf5"])
def test_completed_processed_outputs_are_verified_skipped_and_never_overwritten(
    request_data: MigrationRequest, mapping: MigrationMapping, relative: str
) -> None:
    source = _legacy_collection(request_data, False)
    with h5py.File(source, "a") as handle:
        handle["unknown"] = np.array([3, 7], dtype=np.int16)
    first = _convert(request_data, mapping)
    root = first.destination.result_path / "data/fluxdep"
    targets = {root / "spectrums.hdf5", root / "spectrums.legacy.hdf5"}
    before = {target: (target.read_bytes(), target.stat().st_mtime_ns) for target in targets}
    report = _convert(replace(request_data, resume=True), mapping)
    assert targets <= {row.destination for row in report.skipped}
    assert {target: (target.read_bytes(), target.stat().st_mtime_ns) for target in targets} == before
    target = root / relative
    target.write_bytes(b"changed binary")
    edited = target.read_bytes()
    manifest = first.destination.result_path / "records/migration-state.json"
    manifest_before = manifest.read_bytes()
    with pytest.raises(MigrationInputError) as caught:
        _convert(replace(request_data, resume=True), mapping)
    assert any(row.destination == target and row.comparison == "sha256" for row in caught.value.differences)
    assert target.read_bytes() == edited
    assert manifest.read_bytes() == manifest_before


def test_processed_spectrum_dry_run_plans_conversion_without_writes(
    request_data: MigrationRequest, mapping: MigrationMapping
) -> None:
    source = _legacy_collection(request_data, False)
    original = source.read_bytes()
    report = _convert(replace(request_data, dry_run=True), mapping)
    assert any(
        item.old_file == source
        and item.source_keys == ("scan.with.dots", "mA_c")
        and item.action == "value"
        for item in report.key_mappings
    ), "Dry-run must report the formal calibration projection"
    assert source.read_bytes() == original
    assert not report.destination.result_path.exists()
    assert report.backup_path is not None
    assert not report.backup_path.exists()


def test_unknown_spectrum_content_is_preserved_without_duplicate_converted_data(
    request_data: MigrationRequest, mapping: MigrationMapping
) -> None:
    source = _legacy_collection(request_data, False)
    with h5py.File(source, "a") as handle:
        handle.attrs["origin"] = "synthetic"
        group = _group(handle, "scan.with.dots")
        group.attrs["operator.note"] = "unchanged"
        _dataset(group, "spectrum/data").attrs["gain"] = np.int16(7)
        group["unknown.values"] = np.array([3, 8], dtype=np.int16)
    report = _convert(request_data, mapping)
    companion = report.destination.result_path / "data/fluxdep/spectrums.legacy.hdf5"
    assert companion.is_file(), (
        "Unsupported HDF5 leaves must be visible in the new entry"
    )
    with h5py.File(companion, "r") as handle:
        assert handle.attrs["origin"] == "synthetic"
        group = _group(handle, "scan.with.dots")
        assert group.attrs["operator.note"] == "unchanged"
        np.testing.assert_array_equal(_dataset(group, "unknown.values")[()], [3, 8])
        assert _dataset(group, "unknown.values").dtype == np.dtype("int16")
        assert _group(group, "spectrum/data").attrs["gain"] == 7
        assert set(group) == {"spectrum", "unknown.values"}
        assert "type" not in group.attrs
    preserved = [item for item in report.key_mappings if item.action == "preserved"]
    row = next(
        item
        for item in preserved
        if item.source_keys == ("scan.with.dots", "unknown.values")
    )
    assert row.destination_keys == ("scan.with.dots", "unknown.values")
    assert row.new_file == companion
    assert row.new_path is None
    assert not any(item.source == source for item in report.pending)
    target = report.destination.result_path / "data/fluxdep/spectrums.hdf5"
    assert load_spectrums(str(target))["scan.with.dots"].get("type") == "TwoTone"
    before = companion.read_bytes()
    resumed = _convert(replace(request_data, resume=True), mapping)
    assert resumed.key_mappings == report.key_mappings
    assert companion.read_bytes() == before


def test_square_axis_without_version_evidence_is_preserved_and_pending(
    request_data: MigrationRequest, mapping: MigrationMapping
) -> None:
    source = _legacy_collection(request_data, False)
    with h5py.File(source, "a") as handle:
        spectrum = _group(handle, "scan.with.dots/spectrum")
        del spectrum["fpts"]
        del spectrum["data"]
        spectrum["fpts"] = np.array([4.0, 5.0])
        spectrum["data"] = np.array([[1j, 2j], [3j, 4j]], dtype=np.complex64)
    report = _convert(request_data, mapping)
    companion = report.destination.result_path / "data/fluxdep/spectrums.legacy.hdf5"
    assert companion.is_file(), (
        "Ambiguous original data must remain visible, not only in backup"
    )
    with h5py.File(companion, "r") as handle:
        np.testing.assert_array_equal(
            _dataset(handle, "scan.with.dots/spectrum/data")[()], [[1j, 2j], [3j, 4j]]
        )
    assert report.backup_path is not None
    backup_source = report.backup_path / "result/chip/qubit/data/fluxdep/spectrums.hdf5"
    assert any(
        item.source == backup_source and "axis" in item.reason.lower()
        for item in report.pending
    )
    assert not (report.destination.result_path / "data/fluxdep/spectrums.hdf5").exists()


def test_each_ambiguous_group_has_a_located_pending_reason(
    request_data: MigrationRequest, mapping: MigrationMapping
) -> None:
    source = _legacy_collection(request_data, False)
    with h5py.File(source, "a") as handle:
        spectrum = _group(handle, "scan.with.dots/spectrum")
        del spectrum["fpts"]
        spectrum["fpts"] = np.array([4.0, 5.0])
        handle.copy("scan.with.dots", "second.scan")
    report = _convert(replace(request_data, part="data"), mapping)
    reasons = " ".join(item.reason for item in report.pending)
    assert "scan.with.dots" in reasons, (
        "Pending must not lose the first ambiguous group"
    )
    assert "second.scan" in reasons


def test_processed_spectrum_publication_rejects_user_edits_on_resume(
    request_data: MigrationRequest, mapping: MigrationMapping
) -> None:
    _legacy_collection(request_data, False)
    request = replace(request_data, part="data")
    report = _convert(request, mapping)
    assert report.converted_files, (
        "Processed collection needs an owned conversion publication"
    )
    target = report.destination.result_path / "data/fluxdep/spectrums.hdf5"
    with h5py.File(target, "a") as handle:
        handle.attrs["caller_edit"] = "keep"
    edited = target.read_bytes()
    with pytest.raises(MigrationInputError, match="hash"):
        _convert(replace(request, resume=True), mapping)
    assert target.read_bytes() == edited
