"""Independent Setup/Soc control facets: public routing and outcome contracts."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryInfo,
    EntryRequest,
    PointCreateRequest,
    PointUseRequest,
    SavedPoint,
    WorkingPointInfo,
)
from zcu_tools.gui.session.device_control import DeviceControlFacet
from zcu_tools.gui.session.device_models import DeviceEntry
from zcu_tools.gui.session.services.connection import (
    ConnectRemoteRequest,
    ConnectRequest,
    SoCConnectionService,
)
from zcu_tools.gui.session.services.project_settings import (
    ConnectionPreferences,
    ProjectSettingsService,
    SetupPreferences,
)
from zcu_tools.gui.session.setup_control import SetupControlFacet, SocControlFacet
from zcu_tools.gui.session.types import SocCfgHandle
from zcu_tools.program.v2.mocksoc import make_mock_soccfg

from tests.gui._control_fakes import CallLog, RecordedCall, RecordingSignal, call
from tests.gui.app.measure.remote._helpers import Fixture


class RecordingSettings(ProjectSettingsService):
    def __init__(self, log: CallLog) -> None:
        self._log = log
        self.preferences = SetupPreferences(entry_name="chip", ip="10.0.0.2", port=7000)
        self.info = EntryInfo(
            "entry", "chip", Path("/tmp/result/chip"), Path("/tmp/Database/chip")
        )

    def get_setup_preferences(self) -> SetupPreferences:
        self._log.add("settings", "get_setup_preferences")
        return self.preferences

    def list_entries(self, *, refresh: bool = False) -> tuple[str, ...]:
        self._log.add("settings", "list_entries", refresh=refresh)
        return ("chip",)

    def new_entry(self, request: EntryRequest) -> EntryInfo:
        self._log.add("settings", "new_entry", request)
        return self.info

    def open_entry(self, request: EntryRequest) -> EntryInfo:
        self._log.add("settings", "open_entry", request)
        return self.info

    def remember_connection(self, req: ConnectionPreferences) -> None:
        self._log.add("settings", "remember_connection", req)


class RecordingContext(ContextControlFacet):
    def __init__(self, log: CallLog) -> None:
        self._log = log

    def use_point(self, request: PointUseRequest) -> SavedPoint:
        self._log.add("context", "use_point", request)
        return SavedPoint(entry_id="entry", point=request.label)

    def new_point(self, request: PointCreateRequest) -> SavedPoint:
        self._log.add("context", "new_point", request)
        return SavedPoint(entry_id="entry", point=request.label)

    def list_points(self, *, refresh: bool = False) -> tuple[WorkingPointInfo, ...]:
        self._log.add("context", "list_points", refresh=refresh)
        return (WorkingPointInfo("base", ""),)

    def get_entry_info(self) -> EntryInfo:
        self._log.add("context", "get_entry_info")
        return EntryInfo(
            "entry", "chip", Path("/tmp/result/chip"), Path("/tmp/Database/chip")
        )


class RecordingConnection(SoCConnectionService):
    def __init__(self, log: CallLog) -> None:
        self._log = log
        self.cfg = make_mock_soccfg()
        self.connection_finished = RecordingSignal(log, "connection_finished")
        self.connection_failed = RecordingSignal(log, "connection_failed")

    def start_connect(self, req: ConnectRequest) -> int:
        self._log.add("connection", "start_connect", req)
        return 7

    def get_soccfg(self) -> SocCfgHandle:
        self._log.add("connection", "get_soccfg")
        return self.cfg


class RecordingDevice(DeviceControlFacet):
    def __init__(self, log: CallLog) -> None:
        self._log = log
        self.entry = DeviceEntry(name="flux", type_name="YOKOGS200", status="connected")

    def list_devices(self) -> list[DeviceEntry]:
        self._log.add("device", "list_devices")
        return [self.entry]

    def get_device_unit(self, name: str) -> str:
        self._log.add("device", "get_device_unit", name)
        return "V"


def _facet() -> tuple[
    SetupControlFacet, SocControlFacet, CallLog, BaseEventBus, RecordingConnection
]:
    log, bus = CallLog(), BaseEventBus()
    connection = RecordingConnection(log)
    return (
        SetupControlFacet(
            bus=bus,
            settings=RecordingSettings(log),
            context=RecordingContext(log),
            device=RecordingDevice(log),
        ),
        SocControlFacet(
            bus=bus, settings=RecordingSettings(log), connection=connection
        ),
        log,
        bus,
        connection,
    )


def test_setup_control_facet_forwards_deliberate_setup_dialog_contract() -> None:
    facet, _soc, log, bus, _connection = _facet()
    request = EntryRequest("chip")
    point = PointCreateRequest("new", "", CommandOrigin("gui"), clone_from="base")
    saved = SavedPoint(entry_id="entry", point="base")
    info = EntryInfo(
        "entry", "chip", Path("/tmp/result/chip"), Path("/tmp/Database/chip")
    )
    cases: tuple[tuple[str, Callable[[], object], object, RecordedCall], ...] = (
        (
            "get_setup_preferences",
            facet.get_setup_preferences,
            SetupPreferences(entry_name="chip", ip="10.0.0.2", port=7000),
            call("settings", "get_setup_preferences"),
        ),
        (
            "list_entries",
            facet.list_entries,
            ("chip",),
            call("settings", "list_entries", refresh=False),
        ),
        (
            "new_entry",
            lambda: facet.new_entry(request),
            info,
            call("settings", "new_entry", request),
        ),
        (
            "open_entry",
            lambda: facet.open_entry(request),
            info,
            call("settings", "open_entry", request),
        ),
        (
            "get_entry_info",
            facet.get_entry_info,
            info,
            call("context", "get_entry_info"),
        ),
        (
            "use_point",
            lambda: facet.use_point(PointUseRequest("base")),
            saved,
            call("context", "use_point", PointUseRequest("base")),
        ),
        (
            "new_point",
            lambda: facet.new_point(point),
            SavedPoint(entry_id="entry", point="new"),
            call("context", "new_point", point),
        ),
        (
            "list_points",
            facet.list_points,
            (WorkingPointInfo("base", ""),),
            call("context", "list_points", refresh=False),
        ),
    )
    assert facet.get_bus() is bus
    for name, action, expected, _call in cases:
        assert action() == expected, name
    assert log.calls == [expected_call for *_, expected_call in cases]


def test_soc_control_forwards_connection_contract() -> None:
    _setup, facet, log, bus, connection = _facet()
    prefs = ConnectionPreferences(ip="127.0.0.1", port=8887)
    request = ConnectRemoteRequest(ip="192.0.2.1", port=8888)
    assert facet.get_bus() is bus
    assert facet.get_setup_preferences() == SetupPreferences(
        entry_name="chip", ip="10.0.0.2", port=7000
    )
    assert facet.start_connect(request) == 7
    assert facet.remember_connection(prefs) is None
    assert facet.get_soccfg() is connection.cfg
    assert log.calls == [
        call("settings", "get_setup_preferences"),
        call("connection", "start_connect", request),
        call("settings", "remember_connection", prefs),
        call("connection", "get_soccfg"),
    ]


def test_setup_control_project_applied_hook_receives_resolved_project(
    qapp, tmp_path
) -> None:
    """Entry publication notifies the current binding owner through its public event."""
    from zcu_tools.gui.session.events import ContextSwitchedPayload

    fixture = Fixture(project_root=str(tmp_path), empty_project=True)
    seen: list[ContextSwitchedPayload] = []
    subscription = fixture.bus.subscribe(ContextSwitchedPayload, seen.append)
    try:
        info = fixture.ctrl.setup_control.new_entry(EntryRequest("chip"))
        assert info.name == "chip"
        assert len(seen) == 1
        assert seen[0].binding_id is not None
        assert fixture.ctrl.context_control.get_source_binding() == seen[0].sources
        assert fixture.ctrl.setup_control.get_entry_info() == info
    finally:
        subscription.unsubscribe()
        fixture.stop()


def test_soc_control_rebinds_connection_outcome_subscription() -> None:
    from zcu_tools.gui.session.events import ConnectionFinishedPayload

    _setup, facet, _log, bus, _connection = _facet()
    first: list[str] = []
    second: list[str] = []
    facet.bind_connection_outcome(
        lambda: first.append("done"), lambda message: first.append(message)
    )
    bus.emit(ConnectionFinishedPayload(success=True, is_mock=True))
    bus.emit(
        ConnectionFinishedPayload(
            success=False, is_mock=True, error_message="first failed"
        )
    )
    facet.bind_connection_outcome(
        lambda: second.append("done"), lambda message: second.append(message)
    )
    bus.emit(ConnectionFinishedPayload(success=True, is_mock=False))
    assert first == ["done", "first failed"]
    assert second == ["done"]
