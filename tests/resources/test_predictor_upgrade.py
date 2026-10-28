"""Explicit predictor file upgrades leave resource folders free of lock files."""

import json
from pathlib import Path

from zcu_tools.resources.predictor_file import load_predictor_file
from zcu_tools.resources.predictor_upgrade import upgrade_predictor_files


def test_apply_preserves_backup_and_writes_without_data_sidecars(
    tmp_path: Path,
) -> None:
    path = tmp_path / "predictor.json"
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

    (result,) = upgrade_predictor_files((path,), apply=True)

    assert result.status == "applied"
    assert result.backup_path is not None
    assert result.backup_path.read_bytes() == original
    assert load_predictor_file(path).flux.offset == 1.0
    assert not list(path.parent.glob("*.lock"))
