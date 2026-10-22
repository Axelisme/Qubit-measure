"""Explicit batch container upgrades through the migrations public seam."""

from pathlib import Path

import h5py
import pytest
from zcu_tools import datafile

import migrations


def _legacy(path: Path) -> None:
    with h5py.File(path, "w") as file:
        file.attrs.update(
            format="zcu.experiment-data",
            format_version="2.1",
            run_id="historic",
            experiment="synthetic",
            started_at="",
            finished_at="",
            completion="unknown",
            labber_path="",
        )
        cfg = file.create_dataset("cfg", data='{"gain": 7}')
        cfg.attrs.update(cfg_type="SyntheticCfg", cfg_schema_version="1.0")
        file.create_dataset("context", data='{"point": "P"}')
        file.create_group("provenance").attrs.update(
            software_versions="{}", git_dirty="unknown"
        )
        file.create_group("data").attrs.update(representation="grouped", metadata="{}")


def test_upgrade_batch_preserves_sources_and_returns_exact_outputs(
    tmp_path: Path,
) -> None:
    sources = [tmp_path / "first.h5", tmp_path / "second.h5"]
    for source in sources:
        _legacy(source)
    before = [path.read_bytes() for path in sources]
    output = tmp_path / "converted"
    actual = migrations.upgrade_containers(sources, output_dir=output)
    assert actual == (output / "first.h5", output / "second.h5")
    assert [path.read_bytes() for path in sources] == before
    for path in actual:
        loaded = datafile.load(path)
        assert loaded.attrs.format_version == "3.0"
        assert loaded.docs["cfg"]["values"] == {"gain": 7}
    with pytest.raises(FileExistsError):
        migrations.upgrade_containers(sources, output_dir=output)


def test_upgrade_cli_runs_in_process_and_rejects_colliding_names(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.h5"
    _legacy(source)
    assert (
        migrations.upgrade_main([str(source), "--output-dir", str(tmp_path / "output")])
        == 0
    )
    second_dir = tmp_path / "other"
    second_dir.mkdir()
    second = second_dir / source.name
    _legacy(second)
    with pytest.raises(ValueError, match="duplicate|collision"):
        migrations.upgrade_containers(
            [source, second], output_dir=tmp_path / "collision"
        )
