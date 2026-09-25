"""Shared NDJSON endpoint port binding behavior."""

from __future__ import annotations

import socket

import pytest


def test_ndjson_endpoint_bind_failure_raises_runtime_error(qapp) -> None:
    """NdjsonRpcEndpoint.start() wraps OSError into RuntimeError with a message.

    We bind a real ephemeral port first, then try to bind it again; the second
    bind raises OSError (EADDRINUSE) which the endpoint must wrap into RuntimeError.
    """
    import socket as socket_mod

    from zcu_tools.gui.remote.rpc_endpoint import ControlOptions, NdjsonRpcEndpoint

    # Hold an ephemeral port open so the second bind fails.
    # listen() is required on Linux: SO_REUSEADDR alone allows a second bind(),
    # but a listening socket makes the second bind raise EADDRINUSE.
    blocker = socket_mod.socket(socket_mod.AF_INET, socket_mod.SOCK_STREAM)
    blocker.setsockopt(socket_mod.SOL_SOCKET, socket_mod.SO_REUSEADDR, 1)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    taken_port = blocker.getsockname()[1]

    class _FakeRouter:
        def on_client_open(self, link):
            pass

        def on_client_close(self, link, *, on_owner_thread):
            pass

        def route(self, link, request):
            pass

    opts = ControlOptions(port=taken_port)

    endpoint = NdjsonRpcEndpoint(
        opts,
        wire_version=1,
        gui_version=1,
        server_name="test_server",
        router=_FakeRouter(),
    )

    try:
        with pytest.raises(RuntimeError, match="bind") as exc_info:
            endpoint.start()

        # The message must name the host:port so the user can act on it.
        msg = str(exc_info.value).lower()
        assert "bind" in msg
        assert str(taken_port) in str(exc_info.value)
    finally:
        blocker.close()


# ---------------------------------------------------------------------------
# 4. Ephemeral fallback (default port) vs fast-fail (pinned port)
# ---------------------------------------------------------------------------


class _FakeRouter:
    def on_client_open(self, link):
        pass

    def on_client_close(self, link, *, on_owner_thread):
        pass

    def route(self, link, request):
        pass


def _take_port() -> tuple[socket.socket, int]:
    """Bind+listen an ephemeral port so a re-bind there raises EADDRINUSE."""
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    return blocker, blocker.getsockname()[1]


def test_default_port_busy_falls_back_to_ephemeral() -> None:
    """allow_port_fallback=True: a taken port retries on 0 and binds elsewhere."""
    from zcu_tools.gui.remote.rpc_endpoint import ControlOptions, NdjsonRpcEndpoint

    blocker, taken_port = _take_port()
    opts = ControlOptions(port=taken_port, allow_port_fallback=True)
    endpoint = NdjsonRpcEndpoint(
        opts,
        wire_version=1,
        gui_version=1,
        server_name="test_fallback",
        router=_FakeRouter(),
    )
    try:
        bound = endpoint.start()
        assert bound != taken_port and bound != 0, (
            "fallback must bind a real OS-assigned port, not the busy one or 0"
        )
        assert endpoint.port == bound
    finally:
        endpoint.stop()
        blocker.close()


def test_pinned_port_busy_fast_fails() -> None:
    """allow_port_fallback=False (explicit --control-port): a taken port raises."""
    from zcu_tools.gui.remote.rpc_endpoint import ControlOptions, NdjsonRpcEndpoint

    blocker, taken_port = _take_port()
    opts = ControlOptions(port=taken_port, allow_port_fallback=False)
    endpoint = NdjsonRpcEndpoint(
        opts,
        wire_version=1,
        gui_version=1,
        server_name="test_pinned",
        router=_FakeRouter(),
    )
    try:
        with pytest.raises(RuntimeError, match="bind"):
            endpoint.start()
    finally:
        blocker.close()
