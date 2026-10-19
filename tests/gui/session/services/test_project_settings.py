"""Entry composition and remembered preferences through public session seams."""

from pathlib import Path

import pytest
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.context_types import (
    EntryRequest,
    PointUseRequest,
    SavedPoint,
)
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.services.project_settings import (
    ConnectionPreferences,
    ProjectSettingsService,
    SetupPreferences,
)
from zcu_tools.gui.session.state import (
    SessionState,
)
from zcu_tools.gui.session.types import ContextReadiness, SessionEnv
from zcu_tools.resources.entry import ResultEntry
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase


@pytest.fixture
def entry_services(
    tmp_path: Path,
) -> tuple[ProjectSettingsService, ContextService, SessionState]:
    state = SessionState(SessionEnv(parameters=None, soc=None, soccfg=None))
    context = ContextService(
        state, BaseEventBus(), bind_assets=ArbWaveformDatabase.info_at
    )
    settings = ProjectSettingsService(
        context,
        state,
        result_root=tmp_path / "result",
        database_root=tmp_path / "Database",
    )
    return settings, context, state


@pytest.mark.parametrize("command", ["new", "open"])
def test_entry_commands_leave_point_unselected_through_real_context_owner(
    tmp_path: Path,
    entry_services: tuple[ProjectSettingsService, ContextService, SessionState],
    command: str,
) -> None:
    settings, context, state = entry_services
    if command == "open":
        ResultEntry.create(
            "entry",
            result_root=tmp_path / "result",
            database_root=tmp_path / "Database",
        )
        info = settings.open_entry(EntryRequest("entry"))
    else:
        info = settings.new_entry(EntryRequest("entry"))

    assert state.session_env.readiness is ContextReadiness.EMPTY
    assert state.session_env.parameters is None
    assert context.get_entry_info() == info
    assert context.list_points() == ()
    assert info.result_path == tmp_path / "result" / "entry"
    assert info.database_path == tmp_path / "Database" / "entry"
    assert state.preferences.entry_name == "entry"
    assert settings.list_entries() == ("entry",)


@pytest.mark.parametrize("command", ["open", "new"])
def test_first_discovery_is_complete_when_entry_command_runs_before_listing(
    tmp_path: Path,
    entry_services: tuple[ProjectSettingsService, ContextService, SessionState],
    command: str,
) -> None:
    settings, _, _ = entry_services
    for name in ("A", "B"):
        ResultEntry.create(
            name, result_root=tmp_path / "result", database_root=tmp_path / "Database"
        )
    if command == "open":
        settings.open_entry(EntryRequest("A"))
        expected = ("A", "B")
    else:
        settings.new_entry(EntryRequest("C"))
        expected = ("A", "B", "C")
    assert settings.list_entries() == expected
    settings.new_entry(EntryRequest("D"))
    assert settings.list_entries() == (*expected, "D")


def test_open_entry_then_use_point_publishes_active_target(
    tmp_path: Path,
    entry_services: tuple[ProjectSettingsService, ContextService, SessionState],
) -> None:
    settings, context, state = entry_services
    entry = ResultEntry.create(
        "entry",
        result_root=tmp_path / "result",
        database_root=tmp_path / "Database",
    )
    entry.new_point("working", description="saved calibration")
    info = settings.open_entry(EntryRequest("entry"))

    target = context.use_point(PointUseRequest("working"))

    assert target == SavedPoint(entry_id=info.entry_id, point="working")
    assert state.session_env.readiness is ContextReadiness.ACTIVE
    assert context.read_context().target == target
    assert context.list_points()[0].description == "saved calibration"


def test_connection_and_entry_preferences_default_without_opening(
    entry_services: tuple[ProjectSettingsService, ContextService, SessionState],
) -> None:
    settings, _, state = entry_services
    assert settings.get_setup_preferences() == SetupPreferences(
        entry_name="", ip="192.168.10.1", port=8887
    )

    settings.remember_connection(ConnectionPreferences(ip="10.0.0.2", port=1234))

    assert state.session_env.readiness is ContextReadiness.EMPTY
    assert settings.get_setup_preferences() == SetupPreferences(
        entry_name="", ip="10.0.0.2", port=1234
    )


def test_connection_request_validates_port() -> None:
    with pytest.raises(ValueError, match="port"):
        ConnectionPreferences(ip="host", port=0)
