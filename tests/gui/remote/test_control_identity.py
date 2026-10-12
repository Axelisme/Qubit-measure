"""Control authority through public RPC; Fluxdep supplies the app composition."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest
from qtpy.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox
from zcu_tools.gui.event_bus import BaseEventBus, EventOrigin
from zcu_tools.gui.remote.control_service import RemoteControlServiceBase
from zcu_tools.gui.remote.method_spec import MethodSpec, build_method_registry
from zcu_tools.gui.remote.rpc_endpoint import ClientLink, ControlOptions
from zcu_tools.gui.remote.wire import Request
from zcu_tools.gui.session.operation_handles import OperationHandles, OperationOutcome

from tests.gui.app.fluxdep.remote._route_harness import (
    ImmediateOwnerScheduler,
    RouteHarness,
    RouteReply,
)
from tests.gui.app.fluxdep.remote._search_harness import PumpedRouteHarness


def raw_client(harness: RouteHarness) -> ClientLink:
    link = ClientLink("control-test", token_required=False)
    harness.service.on_client_open(link)
    harness.links.append(link)
    return link


def result(reply: RouteReply) -> dict[str, object]:
    assert reply["ok"] is True, reply
    return reply["result"]


def reason(reply: RouteReply, expected: str) -> None:
    assert reply["ok"] is False, reply
    assert reply["error"].get("reason") == expected


@pytest.fixture
def control_rpc(tmp_path: Path) -> Iterator[RouteHarness]:
    harness = RouteHarness.create(str(tmp_path))
    try:
        yield harness
    finally:
        harness.close()


def test_uninitialized_socket_reads_but_cannot_write_or_manage_control(control_rpc):
    h = control_rpc
    assert result(h.request(raw_client(h), "control.status")) == {
        "holder": "gui",
        "locked": False,
        "online": True,
        "me": False,
    }
    link = raw_client(h)
    original = result(h.request(link, "project.info"))
    reply = h.request(link, "project.setup", chip_name="changed", qub_name="q")
    reason(reply, "not_holder")
    assert not reply["ok"]
    assert reply["error"]["data"] == {"holder": "gui", "locked": False, "online": True}
    assert result(h.request(link, "project.info")) == original
    for method in ("control.acquire", "control.release"):
        reason(h.request(link, method), "readonly_connection")


@pytest.mark.parametrize("dialog_type", [QDialog, QFileDialog, QMessageBox])
@pytest.mark.parametrize("command", ["connect", "acquire"])
def test_local_modal_refuses_owner_until_closed(
    control_rpc, qapp, dialog_type, command
):
    h = control_rpc
    link = raw_client(h)
    original = result(h.request(link, "project.info"))
    if command == "acquire":
        result(h.request(link, "control.connect", label="writer"))
        result(h.request(link, "control.release"))
    dialog = dialog_type()
    if isinstance(dialog, QFileDialog):
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog)
    dialog.setModal(True)
    dialog.show()
    try:
        assert QApplication.activeModalWidget() is dialog
        params = {"label": "writer"} if command == "connect" else {}
        reply = h.request(link, f"control.{command}", **params)
        reason(reply, "local_busy")
        assert reply["error"]["message"] == "Local 使用中"
        assert result(h.request(link, "control.status"))["holder"] == "gui"
        reason(
            h.request(link, "project.setup", chip_name="bad", qub_name="bad"),
            "not_holder",
        )
        assert result(h.request(link, "project.info")) == original
        reader = raw_client(h)
        assert (
            result(h.request(reader, "control.connect", mode="readonly"))["holder"]
            == "gui"
        )
        reason(h.request(reader, "control.acquire"), "readonly_connection")
    finally:
        dialog.reject()
    assert QApplication.activeModalWidget() is None
    if command == "connect":
        result(h.request(link, "control.connect", label="writer"))
    else:
        result(h.request(link, "control.acquire"))
    result(h.request(link, "project.setup", chip_name="new", qub_name="q"))
    assert result(h.request(link, "project.info"))["chip_name"] == "new"


def test_remote_modal_allows_same_holder_reconnect_and_queries(control_rpc, qapp):
    h = control_rpc
    link = raw_client(h)
    result(h.request(link, "control.connect", label="same"))
    dialog = QDialog()
    dialog.setModal(True)
    dialog.show()
    try:
        assert result(h.request(link, "control.connect", label="same"))["me"] is True
        reconnect = raw_client(h)
        assert (
            result(h.request(reconnect, "control.connect", label="same"))["me"] is True
        )
        assert result(h.request(reconnect, "control.acquire"))["me"] is True
        assert (
            result(h.request(raw_client(h), "control.status"))["holder"] == "rpc:same"
        )
    finally:
        dialog.reject()


def test_holder_writes_without_reading_business_state(control_rpc):
    h = control_rpc
    link = raw_client(h)
    assert result(h.request(link, "control.connect", label="writer"))["me"] is True
    result(h.request(link, "project.setup", chip_name="new-chip", qub_name="new-qubit"))
    project = result(h.request(link, "project.info"))
    assert project["chip_name"] == "new-chip"
    assert project["qub_name"] == "new-qubit"


def test_connect_defaults_fixed_identity_and_idempotency(control_rpc):
    h = control_rpc
    assert result(h.request(raw_client(h), "control.status")) == {
        "holder": "gui",
        "locked": False,
        "online": True,
        "me": False,
    }
    link = raw_client(h)
    remote = {"holder": "rpc:agent", "locked": False, "online": True, "me": True}
    assert result(h.request(link, "control.connect")) == remote
    assert (
        result(h.request(link, "control.connect", label="agent", mode="owner"))
        == remote
    )
    for params in ({"label": "other"}, {"mode": "readonly"}):
        reply = h.request(link, "control.connect", **params)
        assert not reply["ok"]
        assert reply["error"]["code"] == "invalid_params"
        assert result(h.request(link, "control.status")) == remote
    result(h.request(link, "control.release"))
    # An idempotent initialization must not reacquire after release.
    assert result(h.request(link, "control.connect"))["holder"] == "gui"


def test_online_holder_same_label_and_offline_takeover(control_rpc):
    h = control_rpc
    assert result(h.request(raw_client(h), "control.status")) == {
        "holder": "gui",
        "locked": False,
        "online": True,
        "me": False,
    }
    a, same, b = raw_client(h), raw_client(h), raw_client(h)
    result(h.request(a, "control.connect", label="A"))
    assert result(h.request(same, "control.connect", label="A"))["me"] is True
    assert result(h.request(b, "control.connect", label="B"))["me"] is False
    reason(h.request(b, "control.acquire"), "control_held")
    reason(h.request(b, "control.release"), "not_holder")
    h.disconnect(a)
    assert result(h.request(b, "control.status"))["online"] is True
    h.disconnect(same)
    assert result(h.request(b, "control.status")) == {
        "holder": "rpc:A",
        "locked": False,
        "online": False,
        "me": False,
    }
    assert result(h.request(b, "control.acquire"))["holder"] == "rpc:B"
    h.disconnect(b)
    reconnect = raw_client(h)
    assert result(h.request(reconnect, "control.connect", label="B"))["me"] is True
    assert result(h.request(reconnect, "control.release")) == {
        "holder": "gui",
        "locked": False,
        "online": True,
        "me": False,
    }


def test_gui_lock_reclaim_unlock_and_watch_share_rpc_projection(control_rpc):
    h = control_rpc
    assert result(h.request(raw_client(h), "control.status")) == {
        "holder": "gui",
        "locked": False,
        "online": True,
        "me": False,
    }
    a, b = raw_client(h), raw_client(h)
    owner = h.ctrl.control_identity
    observed = []
    dispose = owner.watch(observed.append)
    owner.toggle_local()
    assert result(h.request(a, "control.connect", label="A"))["locked"] is True
    reason(h.request(a, "control.acquire"), "control_locked")
    owner.toggle_local()
    result(h.request(a, "control.acquire"))
    owner.reclaim_local()
    reason(h.request(a, "control.acquire"), "control_locked")
    assert result(h.request(b, "control.status"))["holder"] == "gui"
    assert observed[-1] == owner.status()
    dispose()
    count = len(observed)
    owner.toggle_local()
    assert len(observed) == count
    assert result(h.request(b, "control.status"))["locked"] is False


def test_readonly_same_label_never_writes_or_acquires(control_rpc):
    h = control_rpc
    assert result(h.request(raw_client(h), "control.status")) == {
        "holder": "gui",
        "locked": False,
        "online": True,
        "me": False,
    }
    a, reader = raw_client(h), raw_client(h)
    result(h.request(a, "control.connect", label="same"))
    assert (
        result(h.request(reader, "control.connect", label="same", mode="readonly"))[
            "me"
        ]
        is False
    )
    original = result(h.request(reader, "project.info"))
    reason(
        h.request(reader, "project.setup", chip_name="bad", qub_name="bad"),
        "not_holder",
    )
    assert result(h.request(reader, "project.info")) == original
    for method in ("control.acquire", "control.release"):
        reason(h.request(reader, method), "readonly_connection")


def test_queued_write_checks_current_holder_at_execution(tmp_path):
    h = PumpedRouteHarness.create(str(tmp_path))
    try:
        a = raw_client(h)
        result(h.request(a, "control.status"))
        result(h.request(a, "control.connect"))
        original = result(h.request(a, "project.info"))
        pending = h.request_async(a, "project.setup", chip_name="bad", qub_name="bad")
        h.ctrl.control_identity.reclaim_local()
        reason(h.wait(pending), "not_holder")
        assert result(h.request(a, "project.info")) == original
    finally:
        h.close()


def test_accepted_operation_survives_transfer_and_disconnect_new_holder_cancels(
    control_rpc,
):
    h = control_rpc
    assert result(h.request(raw_client(h), "control.status")) == {
        "holder": "gui",
        "locked": False,
        "online": True,
        "me": False,
    }
    handles = OperationHandles()
    token = 0

    def start(_adapter: object, _params: Mapping[str, object]) -> dict[str, object]:
        nonlocal token
        token = handles.create(
            cancel_hook=lambda: handles.settle(token, OperationOutcome("cancelled")),
            origin=EventOrigin(kind="user"),
        )
        return {"token": token}

    def cancel(_adapter: object, _params: Mapping[str, object]) -> dict[str, object]:
        handles.cancel(token)
        return {}

    def wait(_adapter: object, _params: Mapping[str, object]) -> dict[str, object]:
        outcome = handles.await_known_outcome(token, 0)
        return {
            "reason": outcome.reason,
            "status": outcome.outcome.status if outcome.outcome else None,
        }

    service = RemoteControlServiceBase(
        SimpleNamespace(bus=BaseEventBus()),
        ControlOptions(port=0),
        owner_scheduler=ImmediateOwnerScheduler(),
        wire_version=1,
        gui_version=1,
        server_name="ControlOperationTest",
        method_registry=build_method_registry(
            {
                "operation.start": start,
                "operation.cancel": cancel,
                "operation.await": wait,
            },
            {
                "operation.start": MethodSpec(1, "Start accepted operation"),
                "operation.cancel": MethodSpec(1, "Cancel accepted operation"),
                "operation.await": MethodSpec(
                    1, "Wait", readonly=True, off_main_thread=True
                ),
            },
        ),
        event_serializers={},
        wire_event_name=str,
        control_identity=h.ctrl.control_identity,
    )
    a, b, reader = [ClientLink("operation", token_required=False) for _ in range(3)]
    for link in (a, b, reader):
        service.on_client_open(link)

    def rpc(link, method, **params):
        service.route(link, Request("op", method, params))
        return json.loads(link.outbound.get_nowait())

    assert rpc(a, "control.connect", label="A")["ok"]
    assert rpc(a, "operation.start")["ok"]
    assert rpc(a, "control.release")["ok"]
    assert rpc(reader, "operation.await")["result"]["reason"] == "timeout"
    service.on_client_close(a, on_owner_thread=True)
    assert rpc(reader, "operation.await")["result"]["reason"] == "timeout"
    assert rpc(b, "control.connect", label="B")["ok"]
    assert rpc(b, "operation.cancel")["ok"]
    assert rpc(reader, "operation.await")["result"] == {
        "reason": "completed",
        "status": "cancelled",
    }
    for link in (b, reader):
        service.on_client_close(link, on_owner_thread=True)
