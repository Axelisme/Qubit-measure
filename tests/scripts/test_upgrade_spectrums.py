"""Public in-process upgrade command behavior for legacy spectrum collections."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from zcu_tools.analysis.fluxdep.io import load_spectrums

from scripts.upgrade_spectrums import main


@pytest.fixture
def legacy_spectrums(tmp_path):
    path = tmp_path / "project" / "result" / "entry" / "spectrums.hdf5"
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as handle:
        group = handle.create_group("scan.with.dots")
        group.attrs["type"] = "OneTone"
        group.create_dataset("flux_half", data=0.25)
        group.create_dataset("flux_int", data=999.0)
        group.create_dataset("flux_period", data=-2.0)
        spect = group.create_group("spectrum")
        spect.create_dataset("dev_values", data=[0.0, 1.0])
        spect.create_dataset("fluxs", data=[0.625, 0.125])
        spect.create_dataset("freqs", data=[4.0])
        spect.create_dataset("signals", data=np.array([[1j], [2j]]))
        points = group.create_group("points")
        for name, values in (
            ("dev_values", [1.0]),
            ("fluxs", [0.125]),
            ("freqs", [4.0]),
        ):
            points.create_dataset(name, data=values)
    return path


def test_preview_then_apply_preserves_arrays_and_backs_up_original_bytes(
    legacy_spectrums, capsys
):
    path = legacy_spectrums
    original = path.read_bytes()
    assert main([str(path)]) == 0
    preview = capsys.readouterr().out
    assert "dry-run" in preview
    assert "/scan.with.dots/offset" in preview
    assert "0.625" in preview
    assert path.read_bytes() == original
    assert list(path.parent.glob("*.bak")) == []

    assert main([str(path), "--apply"]) == 0
    report = capsys.readouterr().out
    assert "applied" in report
    backup_dir = path.parents[2].with_name("project-backup") / "result" / "entry"
    backups = list(backup_dir.glob("*.bak"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == original
    result = load_spectrums(str(path))["scan.with.dots"]
    assert result["offset"] == 0.625
    assert result["period"] == -2.0
    assert result["device"] == "unnamed_device"
    assert result["unit"] == "native"
    assert result.get("type") == "OneTone"
    np.testing.assert_array_equal(result["spectrum"]["fluxs"], [0.625, 0.125])
    np.testing.assert_array_equal(result["points"]["fluxs"], [0.125])
    np.testing.assert_array_equal(result["spectrum"]["signals"], [[1j], [2j]])
    upgraded = path.read_bytes()
    assert main([str(path), "--apply"]) == 0
    assert "skipped" in capsys.readouterr().out
    assert path.read_bytes() == upgraded
    assert list(backup_dir.glob("*.bak")) == backups
    assert list(path.parent.iterdir()) == [path]
    path.write_bytes(backups[0].read_bytes())
    assert path.read_bytes() == original


def test_standalone_requires_external_override_before_writing(
    legacy_spectrums, tmp_path, capsys
):
    source_dir = tmp_path / "standalone"
    source_dir.mkdir()
    path = source_dir / "spectrums.hdf5"
    legacy_spectrums.replace(path)
    original = path.read_bytes()
    before = set(tmp_path.rglob("*"))
    assert main([str(path)]) == 0
    assert main([str(path), "--apply"]) == 1
    assert "--backup-root" in capsys.readouterr().out
    assert (
        main([str(path), "--apply", "--backup-root", str(source_dir / "backups")]) == 1
    )
    assert set(tmp_path.rglob("*")) == before
    assert path.read_bytes() == original
    backup_root = tmp_path / "archive"
    assert main([str(path), "--apply", "--backup-root", str(backup_root)]) == 0
    (backup,) = backup_root.glob("spectrums.hdf5.*.bak")
    path.write_bytes(backup.read_bytes())
    assert path.read_bytes() == original


@pytest.mark.parametrize("period", [0.0, float("nan")])
def test_invalid_source_is_rejected_without_mutation_or_backup(
    legacy_spectrums, period, capsys
):
    with h5py.File(legacy_spectrums, "r+") as handle:
        dataset = handle["scan.with.dots/flux_period"]
        assert isinstance(dataset, h5py.Dataset)
        dataset[()] = period
    original = legacy_spectrums.read_bytes()
    assert main([str(legacy_spectrums), "--apply"]) == 1
    assert "rejected" in capsys.readouterr().out
    assert legacy_spectrums.read_bytes() == original
    assert list(legacy_spectrums.parent.glob("*.bak")) == []
