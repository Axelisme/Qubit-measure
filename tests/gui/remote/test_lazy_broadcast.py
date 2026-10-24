"""Concurrency and backpressure contract for subscriber-aware lazy push."""

from __future__ import annotations

import socket
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

    def register(link: ClientLink) -> socket.socket:
        # Broadcast only uses the socket as a registry key; no network I/O runs.
        sock = MagicMock(spec=socket.socket)
        # tolerated: Synthetic registry setup uses the endpoint's own synchronization.
        with value._clients_lock:  # pyright: ignore[reportPrivateUsage]
            # tolerated: Existing queue/backpressure cases need synthetic recipients without a live writer.
            value._clients[sock] = link  # pyright: ignore[reportPrivateUsage]
        return sock

    yield value, register


def _link(peer: str, *, subscribed: bool = False) -> ClientLink:
    link = ClientLink(peer, token_required=False)
    link.app_ctx = {"subscribed"} if subscribed else set()
    return link


def _matches(link: ClientLink) -> bool:
    subscriptions = link.app_ctx
    if not isinstance(subscriptions, set):
        raise TypeError("Expected a subscription set")
    return "subscribed" in subscriptions


def _queued(link: ClientLink) -> list[bytes]:
    result: list[bytes] = []
    while not link.outbound.empty():
        result.append(link.outbound.get_nowait())
    return result


@pytest.mark.parametrize("registered", [False, True])
def test_zero_matching_recipient_does_not_build(endpoint, registered: bool) -> None:
    endpoint, register = endpoint
    calls = 0
    if registered:
        register(_link("other"))

    def factory() -> bytes:
        nonlocal calls
        calls += 1
        return b"event\n"

    endpoint.broadcast_lazy(factory, _matches)

    assert calls == 0


def test_closing_match_does_not_build(endpoint) -> None:
    endpoint, register = endpoint
    link = _link("closing", subscribed=True)
    link.closing = True
    register(link)
    factory = MagicMock(return_value=b"event\n")

    endpoint.broadcast_lazy(factory, _matches)

    factory.assert_not_called()
    assert _queued(link) == []


def test_multiple_matches_build_once_and_preserve_order(endpoint) -> None:
    endpoint, register = endpoint
    first = _link("first", subscribed=True)
    other = _link("other")
    second = _link("second", subscribed=True)
    for link in (first, other, second):
        register(link)
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
    endpoint, register = endpoint
    link = _link("match", subscribed=True)
    register(link)
    delivered = MagicMock()

    endpoint.broadcast_lazy(lambda: None, _matches, on_delivered=delivered)

    assert _queued(link) == []
    delivered.assert_not_called()


def test_full_slow_queue_does_not_block_healthy_client(endpoint) -> None:
    endpoint, register = endpoint
    slow = _link("slow", subscribed=True)
    healthy = _link("healthy", subscribed=True)
    register(slow)
    register(healthy)
    for _ in range(slow.outbound.maxsize):
        slow.outbound.put_nowait(b"old\n")

    endpoint.broadcast_lazy(lambda: b"new\n", _matches)

    assert slow.consecutive_drops == 1
    assert _queued(healthy) == [b"new\n"]


def test_lazy_broadcast_preserves_per_client_event_order(endpoint) -> None:
    endpoint, register = endpoint
    first = _link("first", subscribed=True)
    second = _link("second", subscribed=True)
    register(first)
    register(second)

    endpoint.broadcast_lazy(lambda: b"a\n", _matches)
    endpoint.broadcast_lazy(lambda: b"b\n", _matches)

    assert _queued(first) == [b"a\n", b"b\n"]
    assert _queued(second) == [b"a\n", b"b\n"]
