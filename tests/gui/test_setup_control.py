"""SetupControlFacet public contract tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.result_scope import ResultScope
from zcu_tools.gui.session.services.connection import ConnectMockRequest
from zcu_tools.gui.session.services.device import DeviceEntry
from zcu_tools.gui.session.services.startup import (
    PersistedStartup,
    ResolvedStartupProject,
    StartupConnectionRequest,
    StartupProjectRequest,
)
from zcu_tools.gui.session.setup_control import SetupControlFacet

from tests.gui._control_fakes import CallLog, RecordedCall, RecordingSignal, call


class RecordingStartup:
    def __init__(self, log: CallLog) -> None:
        self._log = log
        self.persisted = PersistedStartup(
            chip_name="chip", qub_name="qub", res_name="res"
        )
        self.scope = ResultScope(
            scope_id="/tmp/result/chip/qub",
            chip_name="chip",
            qub_name="qub",
            result_dir="/tmp/result/chip/qub",
            params_path="/tmp/result/chip/qub/params.json",
            source="discovered",
        )
        self.resolved = ResolvedStartupProject(
            chip_name="chip",
            qub_name="qub",
            res_name="res",
            result_dir="/tmp/result/chip/qub",
            database_path="/tmp/Database/chip/qub",
            params_path="/tmp/result/chip/qub/params.json",
            scope_id="/tmp/result/chip/qub",
        )

    def get_persisted(self) -> PersistedStartup:
        self._log.add("startup", "get_persisted")
        return self.persisted

    def list_result_scopes(self, *, refresh: bool = False) -> tuple[ResultScope, ...]:
        self._log.add("startup", "list_result_scopes", refresh=refresh)
        return (self.scope,)

    def apply_project(self, req: StartupProjectRequest) -> ResolvedStartupProject:
        self._log.add("startup", "apply_project", req)
        return self.resolved

    def remember_connection(self, req: StartupConnectionRequest) -> None:
        self._log.add("startup", "remember_connection", req)


class RecordingContext:
    def __init__(self, log: CallLog) -> None:
        self._log = log

    def use_context(self, label: str) -> None:
        self._log.add("context", "use_context", label)

    def new_context(
        self,
        bind_device: str | None = None,
        clone_from: str | None = None,
    ) -> None:
        self._log.add(
            "context",
            "new_context",
            bind_device=bind_device,
            clone_from=clone_from,
        )

    def get_context_labels(self) -> list[str]:
        self._log.add("context", "get_context_labels")
        return ["base"]

    def get_active_context_label(self) -> str | None:
        self._log.add("context", "get_active_context_label")
        return "base"


class RecordingConnection:
    def __init__(self, log: CallLog) -> None:
        self._log = log
        self.connection_finished = RecordingSignal(log, "connection_finished")
        self.connection_failed = RecordingSignal(log, "connection_failed")

    def start_connect(self, req: ConnectMockRequest) -> int:
        self._log.add("connection", "start_connect", req)
        return 7

    def get_soccfg(self) -> object:
        self._log.add("connection", "get_soccfg")
        return "cfg"


class RecordingDevice:
    def __init__(self, log: CallLog) -> None:
        self._log = log
        self.entry = DeviceEntry(
            name="flux",
            type_name="YOKOGS200",
            status="connected",
        )

    def list_devices(self) -> list[DeviceEntry]:
        self._log.add("device", "list_devices")
        return [self.entry]

    def get_device_unit(self, name: str) -> str:
        self._log.add("device", "get_device_unit", name)
        return "V"


def _facet(
    *,
    on_project_applied: Callable[[ResolvedStartupProject], None] | None = None,
) -> tuple[SetupControlFacet, CallLog, BaseEventBus, RecordingConnection]:
    log = CallLog()
    bus = BaseEventBus()
    startup = RecordingStartup(log)
    context = RecordingContext(log)
    connection = RecordingConnection(log)
    device = RecordingDevice(log)
    return (
        SetupControlFacet(
            bus=bus,
            startup=cast(Any, startup),
            context=cast(Any, context),
            connection=cast(Any, connection),
            device=cast(Any, device),
            on_project_applied=on_project_applied,
        ),
        log,
        bus,
        connection,
    )


def test_setup_control_facet_forwards_deliberate_setup_dialog_contract() -> None:
    facet, log, bus, _connection = _facet()
    req = StartupProjectRequest("chip", "qub", "res")
    conn_req = StartupConnectionRequest(ip="127.0.0.1", port=8887)
    connect_req = ConnectMockRequest()

    cases: tuple[tuple[str, Callable[[], object], object, RecordedCall], ...] = (
        (
            "get_persisted_startup",
            facet.get_persisted_startup,
            PersistedStartup(chip_name="chip", qub_name="qub", res_name="res"),
            call("startup", "get_persisted"),
        ),
        (
            "list_result_scopes",
            facet.list_result_scopes,
            (
                ResultScope(
                    scope_id="/tmp/result/chip/qub",
                    chip_name="chip",
                    qub_name="qub",
                    result_dir="/tmp/result/chip/qub",
                    params_path="/tmp/result/chip/qub/params.json",
                    source="discovered",
                ),
            ),
            call("startup", "list_result_scopes", refresh=False),
        ),
        (
            "apply_startup_project",
            lambda: facet.apply_startup_project(req),
            True,
            call("startup", "apply_project", req),
        ),
        (
            "use_context",
            lambda: facet.use_context("base"),
            None,
            call("context", "use_context", "base"),
        ),
        (
            "new_context",
            lambda: facet.new_context(bind_device="flux", clone_from="base"),
            None,
            call("context", "new_context", bind_device="flux", clone_from="base"),
        ),
        (
            "get_context_labels",
            facet.get_context_labels,
            ["base"],
            call("context", "get_context_labels"),
        ),
        (
            "get_active_context_label",
            facet.get_active_context_label,
            "base",
            call("context", "get_active_context_label"),
        ),
        (
            "start_connect",
            lambda: facet.start_connect(connect_req),
            7,
            call("connection", "start_connect", connect_req),
        ),
        (
            "remember_startup_connection",
            lambda: facet.remember_startup_connection(conn_req),
            None,
            call("startup", "remember_connection", conn_req),
        ),
        (
            "get_soccfg",
            facet.get_soccfg,
            "cfg",
            call("connection", "get_soccfg"),
        ),
        (
            "list_devices",
            facet.list_devices,
            [DeviceEntry(name="flux", type_name="YOKOGS200", status="connected")],
            call("device", "list_devices"),
        ),
        (
            "get_device_unit",
            lambda: facet.get_device_unit("flux"),
            "V",
            call("device", "get_device_unit", "flux"),
        ),
    )

    assert facet.get_bus() is bus
    for name, action, expected_result, _expected_call in cases:
        assert action() == expected_result, name

    assert log.calls == [expected_call for *_, expected_call in cases]


def test_setup_control_project_applied_hook_receives_resolved_project() -> None:
    seen: list[ResolvedStartupProject] = []
    facet, log, _bus, _connection = _facet(on_project_applied=seen.append)
    req = StartupProjectRequest("chip", "qub", "res")

    assert facet.apply_startup_project(req) is True

    assert log.calls == [call("startup", "apply_project", req)]
    assert len(seen) == 1
    assert seen[0].params_path == "/tmp/result/chip/qub/params.json"


def test_setup_control_rebinds_connection_outcome_signals() -> None:
    facet, log, _bus, connection = _facet()

    def on_finished() -> None:
        raise AssertionError("not called")

    def on_failed(_message: str) -> None:
        raise AssertionError("not called")

    facet.bind_connection_outcome(on_finished, on_failed)

    assert log.calls == [
        call("connection_finished", "disconnect"),
        call("connection_failed", "disconnect"),
        call("connection_finished", "connect", on_finished),
        call("connection_failed", "connect", on_failed),
    ]
    assert connection.connection_finished.handlers == [on_finished]
    assert connection.connection_failed.handlers == [on_failed]
