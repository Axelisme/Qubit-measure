"""Entry composition and remembered preferences through public session seams."""

from pathlib import Path

import pytest
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.context_types import (
    EntryDraft,
    EntryRequest,
    PointUseRequest,
    SavedPoint,
)
from zcu_tools.gui.session.ports import DeviceMemoryInfo
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.services.project_settings import (
    ConnectionPreferences,
    PersistedDeviceEntry,
    PersistedStartup,
    ProjectSettingsService,
    SetupPreferences,
)
from zcu_tools.gui.session.state import (
    DEFAULT_LEFT_PANEL_WIDTH,
    DeviceState,
    DeviceStatus,
    SessionState,
)
from zcu_tools.gui.session.types import ContextReadiness, SessionEnv
from zcu_tools.resources.entry import ResultEntry
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase


class RememberedDevices:
    """Recording adapter for the unrelated remembered-device registration seam."""

    def __init__(self) -> None:
        self.entries: list[DeviceMemoryInfo] = []

    def register_remembered_devices(self, entries: list[DeviceMemoryInfo]) -> None:
        self.entries = list(entries)


@pytest.fixture
def entry_services(
    tmp_path: Path,
) -> tuple[ProjectSettingsService, ContextService, SessionState, RememberedDevices]:
    state = SessionState(SessionEnv(parameters=None, soc=None, soccfg=None))
    context = ContextService(
        state, BaseEventBus(), bind_assets=ArbWaveformDatabase.info_at
    )
    devices = RememberedDevices()
    settings = ProjectSettingsService(
        context,
        devices,
        state,
        result_root=tmp_path / "result",
        database_root=tmp_path / "Database",
    )
    return settings, context, state, devices


@pytest.mark.parametrize("command", ["new", "open"])
def test_entry_commands_publish_draft_through_real_context_owner(
    tmp_path: Path,
    entry_services: tuple[
        ProjectSettingsService, ContextService, SessionState, RememberedDevices
    ],
    command: str,
) -> None:
    settings, context, state, _ = entry_services
    if command == "open":
        ResultEntry.create(
            "entry",
            result_root=tmp_path / "result",
            database_root=tmp_path / "Database",
        )
        info = settings.open_entry(EntryRequest("entry"))
    else:
        info = settings.new_entry(EntryRequest("entry"))

    assert state.session_env.readiness is ContextReadiness.DRAFT
    assert context.read_context().target == EntryDraft(entry_id=info.entry_id)
    assert context.get_entry_info() == info
    assert context.list_points() == ()
    assert info.result_path == tmp_path / "result" / "entry"
    assert info.database_path == tmp_path / "Database" / "entry"
    assert state.preferences.entry_name == "entry"
    assert settings.list_entries() == ("entry",)


@pytest.mark.parametrize("command", ["open", "new"])
def test_first_discovery_is_complete_when_entry_command_runs_before_listing(
    tmp_path: Path,
    entry_services: tuple[
        ProjectSettingsService, ContextService, SessionState, RememberedDevices
    ],
    command: str,
) -> None:
    settings, _, _, _ = entry_services
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
    entry_services: tuple[
        ProjectSettingsService, ContextService, SessionState, RememberedDevices
    ],
) -> None:
    settings, context, state, _ = entry_services
    entry = ResultEntry.create(
        "entry",
        result_root=tmp_path / "result",
        database_root=tmp_path / "Database",
    )
    entry.new_point("working", description="saved calibration")
    info = settings.open_entry(EntryRequest("entry"))

    target = context.use_point(PointUseRequest(info.entry_id, "working"))

    assert target == SavedPoint(entry_id=info.entry_id, point="working")
    assert state.session_env.readiness is ContextReadiness.ACTIVE
    assert context.read_context().target == target
    assert context.list_points()[0].description == "saved calibration"


def test_connection_and_entry_preferences_default_without_opening(
    entry_services: tuple[
        ProjectSettingsService, ContextService, SessionState, RememberedDevices
    ],
) -> None:
    settings, _, state, _ = entry_services
    assert settings.get_setup_preferences() == SetupPreferences(
        entry_name="", ip="192.168.10.1", port=8887
    )

    settings.remember_connection(ConnectionPreferences(ip="10.0.0.2", port=1234))

    assert state.session_env.readiness is ContextReadiness.EMPTY
    assert settings.get_setup_preferences() == SetupPreferences(
        entry_name="", ip="10.0.0.2", port=1234
    )


def test_capture_settings_projects_remembered_devices_and_preferences(
    entry_services: tuple[
        ProjectSettingsService, ContextService, SessionState, RememberedDevices
    ],
) -> None:
    settings, _, state, _ = entry_services
    settings.remember_connection(ConnectionPreferences(ip="host", port=8887))
    for name, remember in (("flux", True), ("probe", True), ("scratch", False)):
        state.put_device(
            DeviceState(
                name=name,
                type_name="FakeDevice",
                address=f"{name}-addr",
                status=DeviceStatus.CONNECTED,
                remember=remember,
            )
        )
    state.preferences.entry_name = "remembered"

    memento = settings.capture_settings(left_panel_width=321)

    assert memento.entry_name == "remembered"
    assert memento.ip == "host"
    assert memento.left_panel_width == 321
    assert tuple(entry.name for entry in memento.devices) == ("flux", "probe")


def test_restore_preferences_and_devices_does_not_open_an_entry(
    entry_services: tuple[
        ProjectSettingsService, ContextService, SessionState, RememberedDevices
    ],
) -> None:
    settings, _, state, devices = entry_services
    assert settings.get_left_panel_width() == DEFAULT_LEFT_PANEL_WIDTH

    settings.restore_settings(
        PersistedStartup(
            entry_name="remembered",
            ip="host",
            port=9999,
            devices=(
                PersistedDeviceEntry(type_name="FakeDevice", name="flux", address="a"),
            ),
            left_panel_width=222,
        )
    )

    assert settings.get_setup_preferences() == SetupPreferences(
        entry_name="remembered", ip="host", port=9999
    )
    assert settings.get_left_panel_width() == 222
    assert devices.entries == [
        DeviceMemoryInfo(type_name="FakeDevice", name="flux", address="a")
    ]
    assert state.session_env.parameters is None
    assert state.session_env.readiness is ContextReadiness.EMPTY


def test_connection_request_validates_port() -> None:
    with pytest.raises(ValueError, match="port"):
        ConnectionPreferences(ip="host", port=0)
