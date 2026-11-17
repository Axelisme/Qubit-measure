"""Concurrency and backpressure contract for subscriber-aware lazy push."""

from __future__ import annotations

import queue
import socket
import threading
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.remote.rpc_endpoint import (
    ClientLink,
    ControlOptions,
    NdjsonRpcEndpoint,
)


class HeldWriterQueue(queue.Queue[bytes]):
    """Keep accepted clients' writer reads behind an explicit teardown release.

    Nonblocking reads still expose the endpoint's enqueued frames to assertions.
    This controls the queue collaborator, not recipient selection or enqueue.
    """

    def __init__(self, maxsize: int, release: threading.Event) -> None:
        super().__init__(maxsize)
        self.release = release

    def get(self, block: bool = True, timeout: float | None = None) -> bytes:
        if block:
            # The fixture releases every writer before stopping/joining the endpoint.
            assert self.release.wait(30), "writer was not released"
        return super().get(block=block, timeout=timeout)


@pytest.fixture()
def endpoint():
    release = threading.Event()
    accepted: list[ClientLink] = []
    sockets: list[socket.socket] = []
    registered = threading.Event()
    router = MagicMock()

    def opened(link: ClientLink) -> None:
        link.outbound = HeldWriterQueue(link.outbound.maxsize, release)
        accepted.append(link)

    router.on_client_open.side_effect = opened
    router.route.side_effect = lambda _link, _request: registered.set()
    value = NdjsonRpcEndpoint(
        ControlOptions(port=0),
        wire_version=1,
        gui_version=1,
        server_name="LazyPushTest",
        router=router,
    )
    value.start()

    def register(prototype: ClientLink) -> ClientLink:
        registered.clear()
        sock = socket.create_connection(("127.0.0.1", value.port))
        sockets.append(sock)
        sock.sendall(b'{"id":"registered","method":"ready","params":{}}\n')
        # Routing a submitted request follows transport registration of this client.
        assert registered.wait(30), "client request was not routed"
        link = accepted[-1]
        link.app_ctx = prototype.app_ctx
        return link

    try:
        yield value, register
    finally:
        release.set()
        value.stop()
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
    link = register(_link("closing", subscribed=True))
    link.closing = True
    factory = MagicMock(return_value=b"event\n")

    endpoint.broadcast_lazy(factory, _matches)

    factory.assert_not_called()
    assert _queued(link) == []


def test_multiple_matches_build_once_and_preserve_order(endpoint) -> None:
    endpoint, register = endpoint
    first = register(_link("first", subscribed=True))
    other = register(_link("other"))
    second = register(_link("second", subscribed=True))
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
    assert delivered == [first.peer, second.peer]
    assert _queued(first) == [b"event\n"]
    assert _queued(other) == []
    assert _queued(second) == [b"event\n"]


def test_none_factory_result_enqueues_nothing_and_has_no_delivery_hook(
    endpoint,
) -> None:
    endpoint, register = endpoint
    link = register(_link("match", subscribed=True))
    delivered = MagicMock()

    endpoint.broadcast_lazy(lambda: None, _matches, on_delivered=delivered)

    assert _queued(link) == []
    delivered.assert_not_called()


def test_full_slow_queue_does_not_block_healthy_client(endpoint) -> None:
    endpoint, register = endpoint
    slow = register(_link("slow", subscribed=True))
    healthy = register(_link("healthy", subscribed=True))
    for _ in range(slow.outbound.maxsize):
        slow.outbound.put_nowait(b"old\n")

    endpoint.broadcast_lazy(lambda: b"new\n", _matches)

    assert slow.consecutive_drops == 1
    assert _queued(healthy) == [b"new\n"]


def test_lazy_broadcast_preserves_per_client_event_order(endpoint) -> None:
    endpoint, register = endpoint
    first = register(_link("first", subscribed=True))
    second = register(_link("second", subscribed=True))

    endpoint.broadcast_lazy(lambda: b"a\n", _matches)
    endpoint.broadcast_lazy(lambda: b"b\n", _matches)

    assert _queued(first) == [b"a\n", b"b\n"]
    assert _queued(second) == [b"a\n", b"b\n"]
