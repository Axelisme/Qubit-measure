"""Concurrency and backpressure contract for subscriber-aware lazy push."""

from __future__ import annotations

import socket
import threading
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from queue import Queue
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
    subscriptions = link.app_ctx
    if not isinstance(subscriptions, set):
        raise TypeError("Expected a subscription set")
    return "subscribed" in subscriptions


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


class _SelectionRouter:
    """Observe real endpoint admission and disconnect without a registry probe."""

    def __init__(self) -> None:
        self.opened: Queue[ClientLink] = Queue()
        self.closed: Queue[ClientLink] = Queue()

    def on_client_open(self, link: ClientLink) -> None:
        link.app_ctx = set()
        self.opened.put(link)

    def on_client_close(self, link: ClientLink, *, on_owner_thread: bool) -> None:
        self.closed.put(link)

    def route(self, link: ClientLink, request: object) -> None:
        raise AssertionError("selection clients send no RPCs")


def _set_subscription(link: ClientLink, *, subscribed: bool) -> None:
    subscriptions = link.app_ctx
    if not isinstance(subscriptions, set):
        raise TypeError("Expected a subscription set")
    if subscribed:
        subscriptions.add("subscribed")
    else:
        subscriptions.discard("subscribed")


@contextmanager
def _selection_endpoint() -> Generator[tuple[NdjsonRpcEndpoint, _SelectionRouter]]:
    router = _SelectionRouter()
    server = NdjsonRpcEndpoint(
        ControlOptions(port=0),
        wire_version=1,
        gui_version=1,
        server_name="selection-test",
        router=router,
    )
    server.start()
    try:
        yield server, router
    finally:
        server.stop()


@contextmanager
def _selected_broadcast(server: NdjsonRpcEndpoint) -> Generator[list[ClientLink]]:
    """Hold the factory after selection; always release and surface worker errors."""
    entered = threading.Event()
    release = threading.Event()
    delivered: list[ClientLink] = []

    def factory() -> bytes:
        entered.set()
        # The context's finally releases the selected factory, even on failure.
        assert release.wait(timeout=30), "selected factory was not released"
        return b"old-event\n"

    with ThreadPoolExecutor(max_workers=1) as pool:
        worker = pool.submit(
            server.broadcast_lazy, factory, _matches, on_delivered=delivered.append
        )
        try:
            # A subscribed real client guarantees the scheduled factory is called.
            assert entered.wait(timeout=30), "recipient selection did not finish"
            yield delivered
        finally:
            release.set()
            # Releasing the factory lets broadcast revalidate and finish.
            worker.result(timeout=30)


@pytest.mark.requires_loopback
def test_unsubscribe_between_selection_and_delivery_prevents_late_push() -> None:
    # Socket timeouts only guard delivery of explicitly enqueued frames.
    with (
        _selection_endpoint() as (server, router),
        socket.create_connection(("127.0.0.1", server.port), timeout=30) as client,
    ):
        link = router.opened.get(timeout=30)  # accept invokes on_client_open
        server.client_state_transaction(
            link, lambda: _set_subscription(link, subscribed=True)
        )
        with _selected_broadcast(server) as delivered:
            server.client_state_transaction(
                link, lambda: _set_subscription(link, subscribed=False)
            )
        assert delivered == []
        server.reply_ok(link, rid="barrier", result={})
        with client.makefile("rb") as reader:
            assert b'"id":"barrier"' in reader.readline()


@pytest.mark.requires_loopback
def test_disconnect_between_selection_and_delivery_prevents_late_push() -> None:
    with (
        _selection_endpoint() as (server, router),
        socket.create_connection(("127.0.0.1", server.port), timeout=30) as client,
    ):
        link = router.opened.get(timeout=30)  # accept invokes on_client_open
        server.client_state_transaction(
            link, lambda: _set_subscription(link, subscribed=True)
        )
        with _selected_broadcast(server) as delivered:
            client.shutdown(socket.SHUT_RDWR)
            # EOF necessarily triggers the router's public cleanup callback.
            assert router.closed.get(timeout=30) is link
        assert delivered == []


@pytest.mark.requires_loopback
def test_subscribe_after_selection_does_not_receive_old_event() -> None:
    with (
        _selection_endpoint() as (server, router),
        socket.create_connection(("127.0.0.1", server.port), timeout=30) as first,
        socket.create_connection(("127.0.0.1", server.port), timeout=30) as second,
    ):
        existing = router.opened.get(timeout=30)  # first accept callback
        late = router.opened.get(timeout=30)  # second accept callback
        server.client_state_transaction(
            existing, lambda: _set_subscription(existing, subscribed=True)
        )
        with _selected_broadcast(server) as delivered:
            server.client_state_transaction(
                late, lambda: _set_subscription(late, subscribed=True)
            )
        assert delivered == [existing]
        server.broadcast_lazy(lambda: b"new-event\n", _matches)
        with first.makefile("rb") as reader:
            assert reader.readline() == b"old-event\n"
            assert reader.readline() == b"new-event\n"
        with second.makefile("rb") as reader:
            assert reader.readline() == b"new-event\n"


def test_delivery_linearizes_before_later_unsubscribe_reply(endpoint) -> None:
    link = _link("match", subscribed=True)
    _register(endpoint, link)

    endpoint.broadcast_lazy(lambda: b"push\n", _matches)

    def unsubscribe() -> None:
        subscriptions = link.app_ctx
        if not isinstance(subscriptions, set):
            raise TypeError("Expected a subscription set")
        subscriptions.discard("subscribed")

    endpoint.client_state_transaction(link, unsubscribe)
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
