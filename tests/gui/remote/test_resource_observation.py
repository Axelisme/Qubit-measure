"""Shared resource observations through the public request/reply seam."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import NotRequired, TypedDict, TypeVar

import pytest
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.remote.control_service import RemoteControlServiceBase
from zcu_tools.gui.remote.method_spec import MethodSpec, build_method_registry
from zcu_tools.gui.remote.observation import ResourceObservationPolicy
from zcu_tools.gui.remote.param_spec import JsonType, ParamSpec
from zcu_tools.gui.remote.rpc_endpoint import ClientLink, ControlOptions
from zcu_tools.gui.remote.wire import Request

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
class ObservedState:
    bus: BaseEventBus = field(default_factory=BaseEventBus)
    versions: dict[str, int] = field(default_factory=lambda: {"data": 0, "aux": 0})
    writes: int = 0
    bad_encoding: bool = False
    failed_read: bool = False
    failed_write: bool = False
    created_identity: str | None = "new"
    created_version: int = 1

    def snapshot(self) -> dict[str, int]:
        return dict(self.versions)


@dataclass
class Harness:
    state: ObservedState
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
    state = ObservedState()
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

    def read_aux(_adapter: object, _params: Mapping[str, object]) -> dict[str, object]:
        return {"aux": state.versions["aux"]}

    def create(_adapter: object, _params: Mapping[str, object]) -> dict[str, object]:
        state.versions["item:new"] = state.created_version
        state.versions["contents:new"] = 1
        return {"identity": state.created_identity}

    optional = (
        ParamSpec("prefix", JsonType.STRING, required=False, default=""),
        ParamSpec("include_full", JsonType.BOOLEAN, required=False, default=False),
    )
    handlers = {
        "read": read,
        "read_aux": read_aux,
        "conditional": read,
        "write": write,
        "aux_write": write,
        "all_write": write,
        "create": create,
        "item_write": write,
        "contents_write": write,
        "read_item": read,
    }
    parameters = {
        "read": optional,
        "conditional": optional,
        "read_item": (ParamSpec("name", JsonType.STRING),),
    }
    specs = {
        name: MethodSpec(0.002, name, params=parameters.get(name, ()))
        for name in handlers
    }
    policies = {
        "read": ResourceObservationPolicy(
            reveals=("data",), reveals_without=("prefix",)
        ),
        "read_aux": ResourceObservationPolicy(reveals=("aux",)),
        "conditional": ResourceObservationPolicy(
            reveals=("data",), reveals_when_nonempty=("include_full",)
        ),
        "write": ResourceObservationPolicy(
            guard_deps=("data",), refresh_after_write=True
        ),
        "aux_write": ResourceObservationPolicy(guard_deps=("aux",)),
        "all_write": ResourceObservationPolicy(guard_deps=("item:*",)),
        "create": ResourceObservationPolicy(
            refresh_after_write=True,
            created_resource="item:{identity}",
            created_identity="identity",
        ),
        "item_write": ResourceObservationPolicy(guard_deps=("item:new",)),
        "contents_write": ResourceObservationPolicy(guard_deps=("contents:new",)),
        "read_item": ResourceObservationPolicy(reveals=("item:{name}",)),
    }
    service = RemoteControlServiceBase(
        state,
        ControlOptions(port=0),
        owner_scheduler=scheduler,
        wire_version=1,
        gui_version=1,
        server_name="ObservationTest",
        method_registry=build_method_registry(handlers, specs),
        event_serializers={},
        wire_event_name=lambda key: str(key),
        resource_versions=state.snapshot,
        observation_policies=policies,
    )
    return Harness(state, scheduler, service)


def _assert_stale(reply: Reply, keys: list[str]) -> None:
    assert not reply["ok"]
    assert "error" in reply
    assert reply["error"]["code"] == "precondition_failed"
    assert "reason" in reply["error"]
    assert "data" in reply["error"]
    assert reply["error"]["reason"] == "stale_version"
    assert reply["error"]["data"] == {"stale": keys}


def test_zero_version_requires_an_explicit_full_read_on_each_connection():
    h = _harness()
    a, b = h.client(), h.client()
    _assert_stale(h.request(a, "write"), ["data"])
    assert h.state.writes == 0
    assert h.request(a, "read")["ok"]
    assert h.request(a, "write")["ok"]
    _assert_stale(h.request(b, "write"), ["data"])
    _assert_stale(h.request(h.client(), "write"), ["data"])
    assert h.state.writes == 1


@pytest.mark.parametrize("prefix", ["", "value"])
def test_explicit_partial_read_does_not_unlock_a_mutation(prefix: str):
    h = _harness()
    link = h.client()
    assert h.request(link, "read", prefix=prefix)["ok"]
    _assert_stale(h.request(link, "write"), ["data"])
    assert h.request(link, "read")["ok"]
    assert h.request(link, "write")["ok"]


def test_conditional_full_read_requires_truthy_original_input():
    h = _harness()
    link = h.client()
    assert h.request(link, "conditional")["ok"]
    _assert_stale(h.request(link, "write"), ["data"])
    assert h.request(link, "conditional", include_full=True)["ok"]
    assert h.request(link, "write")["ok"]


def test_self_write_advances_seen_but_does_not_unlock_unseen_side_effects():
    h = _harness()
    link = h.client()
    h.request(link, "read")
    assert h.request(link, "write")["ok"]
    assert h.request(link, "write")["ok"]
    _assert_stale(h.request(link, "aux_write"), ["aux"])
    h.state.versions["data"] += 1
    _assert_stale(h.request(link, "write"), ["data"])
    assert h.state.writes == 2


def test_self_write_does_not_repair_a_previously_seen_stale_side_effect():
    h = _harness()
    link = h.client()
    h.request(link, "read")
    h.request(link, "read_aux")
    h.state.versions["aux"] += 1
    assert h.request(link, "write")["ok"]
    _assert_stale(h.request(link, "aux_write"), ["aux"])
    assert h.request(link, "write")["ok"]


def test_wildcard_guard_includes_current_and_previously_seen_deleted_keys():
    h = _harness()
    link = h.client()
    h.state.versions.update({"item:a": 1, "item:b": 2})
    _assert_stale(h.request(link, "all_write"), ["item:a", "item:b"])
    h.request(link, "read_item", name="a")
    h.request(link, "read_item", name="b")
    assert h.request(link, "all_write")["ok"]
    del h.state.versions["item:a"]
    _assert_stale(h.request(link, "all_write"), ["item:a"])


def test_creation_certifies_only_the_declared_existence_key():
    h = _harness()
    link = h.client()
    assert h.request(link, "create")["ok"]
    assert h.request(link, "item_write")["ok"]
    _assert_stale(h.request(link, "contents_write"), ["contents:new"])
    h.state.versions["item:new"] = 2
    _assert_stale(h.request(link, "item_write"), ["item:new"])


@pytest.mark.parametrize(("identity", "version"), [("", 1), (None, 1), ("new", 2)])
def test_creation_without_a_valid_identity_and_zero_to_one_change_is_unseen(
    identity: str | None, version: int
):
    h = _harness()
    link = h.client()
    h.state.created_identity, h.state.created_version = identity, version
    assert h.request(link, "create")["ok"]
    _assert_stale(h.request(link, "item_write"), ["item:new"])


@pytest.mark.parametrize("failure", ["handler", "encoding"])
def test_failed_write_keeps_effects_but_does_not_advance_observation(failure: str):
    h = _harness()
    link = h.client()
    h.request(link, "read")
    h.state.failed_write = failure == "handler"
    h.state.bad_encoding = failure == "encoding"
    assert not h.request(link, "write")["ok"]
    assert h.state.writes == 1
    assert h.state.versions["data"] == 1
    h.state.failed_write = h.state.bad_encoding = False
    _assert_stale(h.request(link, "write"), ["data"])
    assert h.state.writes == 1


@pytest.mark.parametrize("prior_read", [False, True])
@pytest.mark.parametrize("failure", ["handler", "encoding"])
def test_failed_read_keeps_missing_or_previous_observation(
    prior_read: bool, failure: str
):
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
    _assert_stale(h.request(link, "write"), ["data"])
    assert h.state.writes == 0


@pytest.mark.uses_wall_clock
def test_timed_out_read_cannot_unlock_write_after_late_owner_delivery():
    h = _harness()
    link = h.client()
    h.scheduler.deferred = True
    reply = h.request(link, "read")
    assert not reply["ok"]
    assert "error" in reply
    assert reply["error"]["code"] == "timeout"
    h.scheduler.flush()
    h.scheduler.deferred = False
    _assert_stale(h.request(link, "write"), ["data"])
    assert h.state.writes == 0
