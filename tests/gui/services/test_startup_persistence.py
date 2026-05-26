from __future__ import annotations

import json
from pathlib import Path

import pytest
from zcu_tools.gui.services.startup_persistence import (
    STARTUP_VERSION,
    PersistedDeviceEntry,
    PersistedStartup,
    StartupPersistenceError,
    StartupPersistenceService,
)


def _settings() -> PersistedStartup:
    return PersistedStartup(
        version=STARTUP_VERSION,
        chip_name="chip",
        qub_name="qubit",
        res_name="res",
        result_dir="/tmp/result",
        database_path="/tmp/database",
        ip="127.0.0.1",
        port=8887,
        devices=[
            PersistedDeviceEntry(type_name="FakeDevice", name="flux", address="none")
        ],
    )


def test_startup_persistence_roundtrip(tmp_path: Path) -> None:
    svc = StartupPersistenceService(cache_dir=tmp_path)

    svc.save(_settings())

    assert svc.load() == _settings()


def test_startup_persistence_rejects_previous_cache_version(tmp_path: Path) -> None:
    svc = StartupPersistenceService(cache_dir=tmp_path)
    path = tmp_path / "startup_v2.json"
    path.write_text(json.dumps({"version": 2, "devices": []}), encoding="utf-8")

    with pytest.raises(StartupPersistenceError, match="Unsupported startup version"):
        svc.load()


def test_startup_persistence_does_not_update_memory_after_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    svc = StartupPersistenceService(cache_dir=tmp_path)

    def fail_replace(self: Path, target: Path) -> Path:
        raise OSError(f"cannot replace {target}")

    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(StartupPersistenceError, match="Failed to save"):
        svc.save(_settings())

    assert svc.get_current().chip_name == ""
