"""Memento DTOs (pydantic v2): JSON round-trip + frozen + version default."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from zcu_tools.gui.app.main.services.persistence_types import (
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


def test_persisted_tab_does_not_carry_save_paths_override():
    # Process-local data/analysis/post path overrides are not persisted; only
    # adapter_name + cfg_raw survive a round-trip. This proves the final
    # contract has no combined save_paths_override projection.
    tab = PersistedTab(adapter_name="fake", cfg_raw={})
    dumped = tab.model_dump(mode="json")
    assert "save_paths_override" not in dumped
    back = PersistedTab.model_validate(dumped)
    assert back.adapter_name == "fake"
    assert back.cfg_raw == {}
