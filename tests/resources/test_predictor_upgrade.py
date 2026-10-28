"""Explicit predictor file upgrades leave resource folders free of lock files."""

import json
from pathlib import Path

import pytest
from zcu_tools.resources.predictor_file import load_predictor_file
from zcu_tools.resources.predictor_upgrade import upgrade_predictor_files


@pytest.mark.parametrize("override", [False, True])
def test_apply_preserves_backup_and_writes_without_data_sidecars(
    tmp_path: Path,
    override: bool,
) -> None:
    path = tmp_path / "project" / "result" / "entry" / "predictors" / "predictor.json"
    path.parent.mkdir(parents=True)
    original = json.dumps(
        {
            "format": "zcu.predictor",
            "version": 1,
            "model": "fluxonium",
            "params": {
                "EJ": 5.0,
                "flux_half": 0.3,
                "flux_period": -0.4,
                "flux_bias": 0.1,
            },
            "flux_unit": "V",
            "source": {"run_id": None, "created_at": None},
        }
    ).encode()
    path.write_bytes(original)

    backup_root = tmp_path / "archive" if override else None
    (result,) = upgrade_predictor_files((path,), apply=True, backup_root=backup_root)

    assert result.status == "applied"
    assert result.backup_path is not None
    assert result.backup_path.read_bytes() == original
    assert (
        result.backup_path.parent
        == (backup_root or tmp_path / "project-backup")
        / "result"
        / "entry"
        / "predictors"
    )
    assert list(path.parent.iterdir()) == [path]
    assert load_predictor_file(path).flux.offset == 1.0
    assert not list(path.parent.glob("*.lock"))
    path.write_bytes(result.backup_path.read_bytes())
    assert path.read_bytes() == original
