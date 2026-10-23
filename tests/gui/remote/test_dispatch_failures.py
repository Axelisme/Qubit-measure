"""Dispatch failures through the public request/reply seam."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import NotRequired, TypedDict, TypeVar

import pytest
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.remote.control_service import RemoteAppSpec, RemoteControlServiceBase
from zcu_tools.gui.remote.method_spec import MethodSpec, build_method_registry
from zcu_tools.gui.remote.rpc_endpoint import ClientLink, ControlOptions, ReplyDelivery
from zcu_tools.gui.remote.wire import Request
from zcu_tools.gui.session.adapters.manual_owner_scheduler import ManualOwnerScheduler

_T = TypeVar("_T")


class ErrorReply(TypedDict):
    code: str
    reason: NotRequired[str]
    data: NotRequired[dict[str, object]]


class Reply(TypedDict):
    id: str
    ok: bool
    result: NotRequired[dict[str, object]]
    error: NotRequired[ErrorReply]


@dataclass
class OwnerScheduler:
    deferred: bool = False
    pending: list[Callable[[], None]] = field(default_factory=list)

    def is_owner_thread(self) -> bool:
        return True

    def post(self, callback: Callable[[], None]) -> None:
        if self.deferred:
            self.pending.append(callback)
        else:
            callback()

    def call(self, callback: Callable[[], _T]) -> _T:
        return callback()

    def flush(self) -> None:
        while self.pending:
            self.pending.pop(0)()


@dataclass
class HandlerState:
    bus: BaseEventBus = field(default_factory=BaseEventBus)
    versions: dict[str, int] = field(default_factory=lambda: {"data": 0, "aux": 0})
    writes: int = 0
    bad_encoding: bool = False
    failed_read: bool = False
    failed_write: bool = False


@dataclass
class Harness:
    state: HandlerState
    scheduler: OwnerScheduler
    service: RemoteControlServiceBase

    def client(self) -> ClientLink:
        link = ClientLink("test-client", token_required=False)
        self.service.on_client_open(link)
        return link

    def request(self, link: ClientLink, method: str, **params: object) -> Reply:
        self.service.route(link, Request("request", method, params))
        reply: Reply = json.loads(link.outbound.get_nowait())
        return reply


def _harness() -> Harness:
    state = HandlerState()
    scheduler = OwnerScheduler()

    def read(_adapter: object, _params: Mapping[str, object]) -> dict[str, object]:
        if state.failed_read:
            raise RuntimeError("read failed")
        return {"value": object() if state.bad_encoding else state.versions["data"]}

    def write(_adapter: object, _params: Mapping[str, object]) -> dict[str, object]:
        state.writes += 1
        state.versions["data"] += 1
        state.versions["aux"] += 1
        if state.failed_write:
            raise RuntimeError("write failed")
        return {"writes": object() if state.bad_encoding else state.writes}

    handlers = {"read": read, "write": write}
    specs = {name: MethodSpec(0.002, name) for name in handlers}
    service = RemoteControlServiceBase(
        state,
        ControlOptions(port=0),
        app=RemoteAppSpec(
            wire_version=1,
            gui_version=1,
            server_name="DispatchFailureTest",
            method_registry=build_method_registry(handlers, specs),
            event_serializers={},
            wire_event_name=lambda key: str(key),
        ),
        owner_scheduler=scheduler,
    )
    return Harness(state, scheduler, service)


@pytest.mark.parametrize("failure", ["handler", "encoding"])
def test_failed_write_keeps_business_effects(failure: str):
    h = _harness()
    link = h.client()
    h.request(link, "read")
    h.state.failed_write = failure == "handler"
    h.state.bad_encoding = failure == "encoding"
    assert not h.request(link, "write")["ok"]
    assert h.state.writes == 1
    assert h.state.versions["data"] == 1


@pytest.mark.parametrize("prior_read", [False, True])
@pytest.mark.parametrize("failure", ["handler", "encoding"])
def test_failed_read_preserves_error_projection(prior_read: bool, failure: str):
    h = _harness()
    link = h.client()
    if prior_read:
        h.request(link, "read")
    h.state.versions["data"] += 1
    h.state.failed_read = failure == "handler"
    h.state.bad_encoding = failure == "encoding"
    reply = h.request(link, "read")
    assert not reply["ok"]
    assert "error" in reply
    assert reply["error"]["code"] == (
        "controller_error" if failure == "handler" else "internal"
    )
    if failure == "encoding":
        assert "reason" in reply["error"]
        assert reply["error"]["reason"] == "response_encoding_failed"


@pytest.mark.parametrize("arrival", ["late", "timeout_boundary"])
@pytest.mark.parametrize("accepted", [True, False])
def test_timed_out_reply_continues_only_accepted_handler_on_owner(
    monkeypatch: pytest.MonkeyPatch, arrival: str, accepted: bool
) -> None:
    scheduler = ManualOwnerScheduler()
    outcomes: list[tuple[ReplyDelivery, bool]] = []
    executions: list[bool] = []

    def handler(_adapter: object, _params: Mapping[str, object]) -> dict[str, object]:
        executions.append(scheduler.is_owner_thread())
        if not accepted:
            raise RuntimeError("shutdown rejected")
        return {"shutting_down": True}

    service = RemoteControlServiceBase(
        HandlerState(),
        ControlOptions(port=0),
        owner_scheduler=scheduler,
        app=RemoteAppSpec(
            wire_version=1,
            gui_version=1,
            server_name="TimeoutContinuationTest",
            method_registry=build_method_registry(
                {"app.shutdown": handler},
                {"app.shutdown": MethodSpec(5.0, "synthetic shutdown")},
            ),
            event_serializers={},
            wire_event_name=str,
            reply_continuations={
                "app.shutdown": lambda outcome: outcomes.append(
                    (outcome, scheduler.is_owner_thread())
                )
            },
        ),
    )
    original_wait = threading.Event.wait

    def expired_wait(event: threading.Event, timeout: float | None = None) -> bool:
        if timeout != 5.0:
            return original_wait(event, timeout)
        if arrival == "timeout_boundary":
            assert scheduler.pump_once(), "the submitted handler must be queued"
        return False

    monkeypatch.setattr(threading.Event, "wait", expired_wait)
    link = ClientLink("timeout-client", token_required=False)
    service.on_client_open(link)
    service.route(link, Request("shutdown", "app.shutdown", {}))
    reply = json.loads(link.outbound.get_nowait())
    assert reply["ok"] is False
    assert reply["error"]["code"] == "timeout"
    assert outcomes == [], "continuation must be posted, not run on the IO caller"
    scheduler.pump_all()
    assert executions == [True]
    assert outcomes == ([(ReplyDelivery.FAILED, True)] if accepted else [])
    scheduler.pump_all()
    assert outcomes == ([(ReplyDelivery.FAILED, True)] if accepted else [])
    assert link.outbound.empty(), "late completion must not send a second RPC reply"


@pytest.mark.uses_wall_clock
def test_timed_out_read_returns_timeout_before_late_owner_delivery():
    h = _harness()
    link = h.client()
    h.scheduler.deferred = True
    reply = h.request(link, "read")
    assert not reply["ok"]
    assert "error" in reply
    assert reply["error"]["code"] == "timeout"
    h.scheduler.flush()
    h.scheduler.deferred = False
