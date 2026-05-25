from __future__ import annotations

from pathlib import Path

import pytest
from zcu_tools.gui.adapter import SavePaths
from zcu_tools.gui.services.session_persistence import (
    PersistedSession,
    PersistedTab,
    SessionPersistenceService,
)


def test_session_persistence_save_and_load_roundtrip(tmp_path: Path):
    svc = SessionPersistenceService(cache_dir=tmp_path)
    session = PersistedSession(
        version=1,
        active_tab_index=0,
        tabs=[
            PersistedTab(
                adapter_name="fake",
                cfg_raw={"x": 1, "sweep": {"start": 0.0, "stop": 1.0, "expts": 11}},
                save_paths_override=SavePaths("/tmp/data.h5", "/tmp/img.png"),
            )
        ],
    )

    svc.save_session(session)
    loaded = svc.load_session()

    assert loaded is not None
    assert loaded.version == 1
    assert loaded.active_tab_index == 0
    assert len(loaded.tabs) == 1
    assert loaded.tabs[0].adapter_name == "fake"
    assert loaded.tabs[0].cfg_raw["x"] == 1
    assert loaded.tabs[0].save_paths_override == SavePaths(
        "/tmp/data.h5", "/tmp/img.png"
    )


def test_session_persistence_missing_file_returns_none(tmp_path: Path):
    svc = SessionPersistenceService(cache_dir=tmp_path)
    assert svc.load_session() is None


def test_session_persistence_restores_legacy_sweep_step_none(tmp_path: Path):
    from zcu_tools.gui.adapter import (
        CfgSchema,
        CfgSectionSpec,
        CfgSectionValue,
        SweepSpec,
    )

    svc = SessionPersistenceService(cache_dir=tmp_path)
    base = CfgSchema(
        spec=CfgSectionSpec(fields={"sweep": SweepSpec(label="Sweep")}),
        value=CfgSectionValue(fields={}),
    )

    restored = svc.raw_to_schema(
        base,
        {"sweep": {"start": 0.0, "stop": 1.0, "expts": 11, "step": None}},
    )
    sweep = restored.value.fields["sweep"]
    assert sweep.step == pytest.approx(0.1)  # type: ignore[union-attr]
