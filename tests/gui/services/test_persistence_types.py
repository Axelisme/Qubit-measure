"""Memento DTOs (pydantic v2): JSON round-trip + frozen + version default."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from zcu_tools.gui.adapter import SavePaths
from zcu_tools.gui.services.persistence_types import (
    APP_STATE_VERSION,
    AppPersistedState,
    PersistedDeviceEntry,
    PersistedSession,
    PersistedStartup,
    PersistedTab,
)


def test_app_state_model_dump_validate_roundtrip():
    state = AppPersistedState(
        startup=PersistedStartup(
            chip_name="chip",
            ip="host",
            port=1234,
            devices=(PersistedDeviceEntry(type_name="T", name="flux", address="a"),),
            left_panel_width=321,
        ),
        session=PersistedSession(
            tabs=(
                PersistedTab(
                    adapter_name="fake",
                    cfg_raw={"x": 1, "nested": {"y": 2}},
                    save_paths_override=SavePaths("d.h5", "i.png"),
                ),
            ),
            active_tab_index=0,
        ),
    )

    dumped = state.model_dump(mode="json")
    assert AppPersistedState.model_validate(dumped) == state


def test_default_version():
    assert AppPersistedState().version == APP_STATE_VERSION


def test_frozen():
    s = PersistedStartup()
    with pytest.raises(ValidationError):
        s.ip = "x"  # type: ignore[misc]


def test_save_paths_override_roundtrips_through_json():
    tab = PersistedTab(
        adapter_name="fake", cfg_raw={}, save_paths_override=SavePaths("d", "i")
    )
    back = PersistedTab.model_validate(tab.model_dump(mode="json"))
    assert back.save_paths_override == SavePaths("d", "i")
