"""Concurrency and backpressure contract for subscriber-aware lazy push."""

from __future__ import annotations

import socket
import threading
from collections.abc import Callable
from typing import cast
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.remote.rpc_endpoint import (
    ClientLink,
    ControlOptions,
    NdjsonRpcEndpoint,
)


@pytest.fixture()
def endpoint():
    value = NdjsonRpcEndpoint(
        ControlOptions(port=0),
        wire_version=1,
        gui_version=1,
        server_name="LazyPushTest",
        router=MagicMock(),
    )
    sockets: list[socket.socket] = []

    def register(link: ClientLink) -> socket.socket:
        sock = socket.socket()
        sockets.append(sock)
        with value._clients_lock:
            value._clients[sock] = link
        return sock

    value.register_test_link = register  # type: ignore[attr-defined]
    yield value
    for sock in sockets:
        sock.close()


def _link(peer: str, *, subscribed: bool = False) -> ClientLink:
    link = ClientLink(peer, token_required=False)
    link.app_ctx = {"subscribed"} if subscribed else set()
    return link


def _matches(link: ClientLink) -> bool:
    return "subscribed" in cast(set[str], link.app_ctx)


def _register(endpoint: NdjsonRpcEndpoint, link: ClientLink) -> socket.socket:
    register = cast(Callable[[ClientLink], socket.socket], endpoint.register_test_link)  # type: ignore[attr-defined]
    return register(link)


def _queued(link: ClientLink) -> list[bytes]:
    result: list[bytes] = []
    while not link.outbound.empty():
        result.append(link.outbound.get_nowait())
    return result


@pytest.mark.parametrize("registered", [False, True])
def test_zero_matching_recipient_does_not_build(endpoint, registered: bool) -> None:
    calls = 0
    if registered:
        _register(endpoint, _link("other"))

    def factory() -> bytes:
        nonlocal calls
        calls += 1
        return b"event\n"

    endpoint.broadcast_lazy(factory, _matches)

    assert calls == 0


def test_closing_match_does_not_build(endpoint) -> None:
    link = _link("closing", subscribed=True)
    link.closing = True
    _register(endpoint, link)
    factory = MagicMock(return_value=b"event\n")

    endpoint.broadcast_lazy(factory, _matches)

    factory.assert_not_called()
    assert _queued(link) == []


def test_multiple_matches_build_once_and_preserve_order(endpoint) -> None:
    first = _link("first", subscribed=True)
    other = _link("other")
    second = _link("second", subscribed=True)
    for link in (first, other, second):
        _register(endpoint, link)
    calls = 0
    delivered: list[str] = []

    def factory() -> bytes:
        nonlocal calls
        calls += 1
        return b"event\n"

    endpoint.broadcast_lazy(
        factory,
        _matches,
        on_delivered=lambda link: delivered.append(link.peer),
    )

    assert calls == 1
    assert delivered == ["first", "second"]
    assert _queued(first) == [b"event\n"]
    assert _queued(other) == []
    assert _queued(second) == [b"event\n"]


def test_none_factory_result_enqueues_nothing_and_has_no_delivery_hook(
    endpoint,
) -> None:
    link = _link("match", subscribed=True)
    _register(endpoint, link)
    delivered = MagicMock()

    endpoint.broadcast_lazy(lambda: None, _matches, on_delivered=delivered)

    assert _queued(link) == []
    delivered.assert_not_called()


def test_unsubscribe_between_selection_and_delivery_prevents_late_push(
    endpoint,
) -> None:
    link = _link("match", subscribed=True)
    _register(endpoint, link)
    factory_entered = threading.Event()
    release_factory = threading.Event()

    def factory() -> bytes:
        factory_entered.set()
        assert release_factory.wait(timeout=2.0)
        return b"old-event\n"

    worker = threading.Thread(
        target=endpoint.broadcast_lazy,
        args=(factory, _matches),
    )
    worker.start()
    assert factory_entered.wait(timeout=2.0)
    endpoint.client_state_transaction(
        link, lambda: cast(set[str], link.app_ctx).discard("subscribed")
    )
    release_factory.set()
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert _queued(link) == []


def test_disconnect_between_selection_and_delivery_prevents_late_push(endpoint) -> None:
    link = _link("match", subscribed=True)
    sock = _register(endpoint, link)
    factory_entered = threading.Event()
    release_factory = threading.Event()

    def factory() -> bytes:
        factory_entered.set()
        assert release_factory.wait(timeout=2.0)
        return b"old-event\n"

    worker = threading.Thread(
        target=endpoint.broadcast_lazy,
        args=(factory, _matches),
    )
    worker.start()
    assert factory_entered.wait(timeout=2.0)
    with endpoint._clients_lock:
        link.closing = True
        endpoint._clients.pop(sock)
    release_factory.set()
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert _queued(link) == []


def test_subscribe_after_selection_does_not_receive_old_event(endpoint) -> None:
    existing = _link("existing", subscribed=True)
    late = _link("late")
    _register(endpoint, existing)
    _register(endpoint, late)
    factory_entered = threading.Event()
    release_factory = threading.Event()

    def factory() -> bytes:
        factory_entered.set()
        assert release_factory.wait(timeout=2.0)
        return b"old-event\n"

    worker = threading.Thread(
        target=endpoint.broadcast_lazy,
        args=(factory, _matches),
    )
    worker.start()
    assert factory_entered.wait(timeout=2.0)
    endpoint.client_state_transaction(
        late, lambda: cast(set[str], late.app_ctx).add("subscribed")
    )
    release_factory.set()
    worker.join(timeout=2.0)

    assert _queued(existing) == [b"old-event\n"]
    assert _queued(late) == []
    endpoint.broadcast_lazy(lambda: b"new-event\n", _matches)
    assert _queued(existing) == [b"new-event\n"]
    assert _queued(late) == [b"new-event\n"]


def test_delivery_linearizes_before_later_unsubscribe_reply(endpoint) -> None:
    link = _link("match", subscribed=True)
    _register(endpoint, link)

    endpoint.broadcast_lazy(lambda: b"push\n", _matches)
    endpoint.client_state_transaction(
        link, lambda: cast(set[str], link.app_ctx).discard("subscribed")
    )
    endpoint.reply_ok(link, rid="unsubscribe", result={"subscribed": []})

    queued = _queued(link)
    assert queued[0] == b"push\n"
    assert b'"id":"unsubscribe"' in queued[1]


def test_full_slow_queue_does_not_block_healthy_client(endpoint) -> None:
    slow = _link("slow", subscribed=True)
    healthy = _link("healthy", subscribed=True)
    _register(endpoint, slow)
    _register(endpoint, healthy)
    for _ in range(slow.outbound.maxsize):
        slow.outbound.put_nowait(b"old\n")

    endpoint.broadcast_lazy(lambda: b"new\n", _matches)

    assert slow.consecutive_drops == 1
    assert _queued(healthy) == [b"new\n"]


def test_lazy_broadcast_preserves_per_client_event_order(endpoint) -> None:
    first = _link("first", subscribed=True)
    second = _link("second", subscribed=True)
    _register(endpoint, first)
    _register(endpoint, second)

    endpoint.broadcast_lazy(lambda: b"a\n", _matches)
    endpoint.broadcast_lazy(lambda: b"b\n", _matches)

    assert _queued(first) == [b"a\n", b"b\n"]
    assert _queued(second) == [b"a\n", b"b\n"]
