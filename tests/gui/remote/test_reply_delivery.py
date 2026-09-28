"""Correlated response delivery through the public endpoint socket contract."""

import json
import queue
import socket
from collections.abc import Iterator
from threading import Event

import pytest
from zcu_tools.gui.remote.errors import ErrorCode
from zcu_tools.gui.remote.framing import MAX_LINE_BYTES, encode_line
from zcu_tools.gui.remote.rpc_endpoint import (
    ClientLink,
    ControlOptions,
    NdjsonRpcEndpoint,
)
from zcu_tools.gui.remote.wire import Request

pytestmark = pytest.mark.requires_loopback


class _RejectingQueue(queue.Queue[bytes]):
    def put_nowait(self, item: bytes) -> None:
        raise queue.Full


class _Router:
    endpoint: NdjsonRpcEndpoint

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.closed = Event()
        self.closed_links: list[ClientLink] = []

    def on_client_open(self, link: ClientLink) -> None:
        pass

    def on_client_close(self, link: ClientLink, *, on_owner_thread: bool) -> None:
        self.closed_links.append(link)
        self.closed.set()

    def route(self, link: ClientLink, request: object) -> None:
        assert isinstance(request, Request)
        self.calls.append(request.method)
        if request.method == "oversized_error":
            self.endpoint.reply_error(
                link,
                rid=request.id,
                code=ErrorCode.CONTROLLER_ERROR,
                message="x" * MAX_LINE_BYTES,
            )
        elif request.method == "large_result":
            self.endpoint.reply_ok(
                link, rid=request.id, result={"value": "x" * (MAX_LINE_BYTES - 1024)}
            )
        elif request.method == "unserializable_error":
            self.endpoint.reply_error(
                link,
                rid=request.id,
                code=ErrorCode.CONTROLLER_ERROR,
                message="handler failed",
                data={"opaque": object()},
            )
        else:
            if request.method == "reject_reply":
                link.outbound = _RejectingQueue()
            value = {
                "oversized_result": "x" * MAX_LINE_BYTES,
                "unserializable_result": object(),
            }.get(request.method, 42)
            self.endpoint.reply_ok(link, rid=request.id, result={"value": value})


@pytest.fixture
def endpoint() -> Iterator[tuple[NdjsonRpcEndpoint, _Router]]:
    router = _Router()
    server = NdjsonRpcEndpoint(
        ControlOptions(port=0),
        wire_version=1,
        gui_version=1,
        server_name="reply-test",
        router=router,
    )
    router.endpoint = server
    server.start()
    try:
        yield server, router
    finally:
        server.stop()
        for link in router.closed_links:
            if link.writer_thread is not None:
                link.writer_thread.join(timeout=2)
                assert not link.writer_thread.is_alive()


def _send(sock: socket.socket, method: str, rid: str = "request") -> None:
    sock.sendall(encode_line({"id": rid, "method": method, "params": {}}))


def _response(sock: socket.socket) -> dict[str, object]:
    with sock.makefile("rb") as reader:
        line = reader.readline(MAX_LINE_BYTES + 2)
    assert line.endswith(b"\n")
    assert len(line) <= MAX_LINE_BYTES + 1
    return json.loads(line)


@pytest.mark.parametrize(
    "method",
    [
        "oversized_result",
        "unserializable_result",
        "oversized_error",
        "unserializable_error",
    ],
)
def test_unencodable_reply_returns_correlated_error_without_replaying_handler(
    endpoint: tuple[NdjsonRpcEndpoint, _Router], method: str
) -> None:
    server, router = endpoint
    with socket.create_connection(("127.0.0.1", server.port), timeout=3) as client:
        _send(client, method)
        reply = _response(client)
        assert reply["id"] == "request"
        assert reply["ok"] is False
        error = reply["error"]
        assert isinstance(error, dict)
        assert error["code"] == "internal"
        assert error["reason"] == "response_encoding_failed"
        assert "may have executed" in error["message"]
        _send(client, "small", "next")
        assert _response(client) == {"id": "next", "ok": True, "result": {"value": 42}}
    assert router.calls == [method, "small"]


@pytest.mark.parametrize("shutdown", [False, True])
def test_backpressured_client_does_not_block_other_clients_or_cleanup(
    endpoint: tuple[NdjsonRpcEndpoint, _Router], shutdown: bool
) -> None:
    server, router = endpoint
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as slow:
        slow.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8192)
        slow.settimeout(3)
        slow.connect(("127.0.0.1", server.port))
        _send(slow, "large_result")
        assert slow.recv(1) == b"{"
        with socket.create_connection(("127.0.0.1", server.port), timeout=3) as healthy:
            _send(healthy, "small")
            assert _response(healthy)["result"] == {"value": 42}
            if shutdown:
                server.stop()
            else:
                slow.close()
            assert router.closed.wait(timeout=3)
    assert router.calls == ["large_result", "small"]


def test_large_reply_backlog_disconnects_slow_client_and_preserves_healthy_client(
    endpoint: tuple[NdjsonRpcEndpoint, _Router],
) -> None:
    server, router = endpoint
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as slow:
        slow.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8192)
        slow.settimeout(3)
        slow.connect(("127.0.0.1", server.port))
        _send(slow, "large_result", "first")
        assert slow.recv(1) == b"{"
        _send(slow, "large_result", "second")
        _send(slow, "large_result", "third")
        assert router.closed.wait(timeout=3)
        with socket.create_connection(("127.0.0.1", server.port), timeout=3) as healthy:
            _send(healthy, "small")
            assert _response(healthy)["result"] == {"value": 42}
    assert router.calls == ["large_result"] * 3 + ["small"]


def test_completed_large_replies_release_outbound_capacity(
    endpoint: tuple[NdjsonRpcEndpoint, _Router],
) -> None:
    server, router = endpoint
    with socket.create_connection(("127.0.0.1", server.port), timeout=3) as client:
        for number in range(4):
            _send(client, "large_result", str(number))
            reply = _response(client)
            assert reply["id"] == str(number)
            assert reply["result"] == {"value": "x" * (MAX_LINE_BYTES - 1024)}
        assert not router.closed.is_set()
    assert router.calls == ["large_result"] * 4


@pytest.mark.parametrize("method", ["unserializable_result", "reject_reply"])
def test_undeliverable_reply_disconnects_and_releases_client_without_stopping_server(
    endpoint: tuple[NdjsonRpcEndpoint, _Router], method: str
) -> None:
    server, router = endpoint
    rid = "x" * (MAX_LINE_BYTES - 80) if method != "reject_reply" else "request"
    with socket.create_connection(("127.0.0.1", server.port), timeout=3) as client:
        _send(client, method, rid)
        assert client.recv(1) == b""
    assert router.closed.wait(timeout=3)
    with socket.create_connection(("127.0.0.1", server.port), timeout=3) as client:
        _send(client, "small")
        assert _response(client)["result"] == {"value": 42}
    assert router.calls == [method, "small"]
