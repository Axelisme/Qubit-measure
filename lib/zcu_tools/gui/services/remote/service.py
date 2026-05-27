"""RemoteControlService — local socket interface for automation agents.

Bind defaults to 127.0.0.1. NDJSON request/response over plain TCP.

Threading:
  - Server thread (daemon): listens, accepts, reads request lines per client,
    parses, looks up the dispatch handler, marshals execution onto the Qt main
    thread, waits (timeout-bounded) for the result, writes one response line.
  - Main thread (Qt): handlers run here via the injected ``marshal_to_main``
    callable. Phase 80 has no per-client writer thread; all writes happen on
    the server thread and are serialised by the in-flight gate (one RPC per
    client at a time).
  - Shutdown is initiated from the main thread (``MainWindow.closeEvent``) and
    wakes the selector loop via a ``socket.socketpair`` self-pipe.
"""

from __future__ import annotations

import hmac
import logging
import selectors
import socket
import threading
from dataclasses import dataclass
from typing import Callable, Optional

from qtpy.QtCore import QObject, Qt, Signal  # type: ignore[attr-defined]

from .dispatch import METHOD_REGISTRY
from .errors import ErrorCode, ErrorEnvelope, RemoteError
from .framing import LINE_TERMINATOR, MAX_LINE_BYTES, decode_line, encode_line
from .wire import Response, _require_str, parse_request

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ControlOptions:
    """Knobs for the remote control endpoint.

    ``port=0`` asks the OS for an ephemeral free port — useful for tests.
    ``allow_external`` flips the bind from loopback to all interfaces; combined
    with a missing ``token`` this is refused at startup.
    """

    port: int
    token: Optional[str] = None
    allow_external: bool = False

    def host(self) -> str:
        return "0.0.0.0" if self.allow_external else "127.0.0.1"


class _MainThreadDispatcher(QObject):
    """QObject living on the Qt main thread.

    Server threads ``emit`` ``invoke`` to schedule a callable on the main thread
    via Qt's queued connection — this is the supported way to marshal arbitrary
    work onto the Qt event loop from a foreign thread (``QTimer.singleShot``
    must run on the timer's own thread).
    """

    invoke = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        # type=QueuedConnection ensures the slot runs on this QObject's owning
        # thread (the Qt main thread) when emitted from a foreign thread.
        self.invoke.connect(self._run, type=Qt.ConnectionType.QueuedConnection)  # type: ignore[call-arg]

    def _run(self, fn: Callable[[], None]) -> None:
        fn()


# ---------------------------------------------------------------------------
# Per-client bookkeeping
# ---------------------------------------------------------------------------


class _ClientState:
    """Tracks per-connection authentication and the incoming-line buffer."""

    __slots__ = ("authed", "buffer", "peer")

    def __init__(self, peer: str, token_required: bool) -> None:
        self.peer = peer
        # If no token is configured, every client starts authenticated.
        self.authed = not token_required
        self.buffer = bytearray()


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class RemoteControlService:
    """Run an NDJSON dispatcher on a local TCP port.

    The caller owns instantiation order: build the Controller / MainWindow
    first, then construct this service with a ``marshal_to_main`` callable that
    schedules a no-arg function on the Qt main thread (typically
    ``lambda fn: QTimer.singleShot(0, fn)``). Service is inert until ``start()``.
    """

    def __init__(
        self,
        controller: object,
        opts: ControlOptions,
    ) -> None:
        if opts.allow_external and not opts.token:
            raise RuntimeError(
                "Remote control must specify a token when --control-allow-external is set"
            )
        self._ctrl = controller
        self._opts = opts
        self._dispatcher = _MainThreadDispatcher()
        self._stopping = threading.Event()
        self._server_sock: Optional[socket.socket] = None
        self._wake_r: Optional[socket.socket] = None
        self._wake_w: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._actual_port: Optional[int] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> int:
        """Bind, listen, spawn the server thread. Returns the actual port."""
        if self._thread is not None:
            raise RuntimeError("RemoteControlService.start() called twice")
        host = self._opts.host()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((host, self._opts.port))
        except OSError as exc:
            sock.close()
            raise RuntimeError(
                f"RemoteControlService bind {host}:{self._opts.port} failed: {exc}"
            ) from exc
        sock.listen(8)
        sock.setblocking(False)
        self._server_sock = sock
        port = int(sock.getsockname()[1])
        self._actual_port = port

        wake_r, wake_w = socket.socketpair()
        wake_r.setblocking(False)
        self._wake_r, self._wake_w = wake_r, wake_w

        self._thread = threading.Thread(
            target=self._serve_forever, name="RemoteControlServer", daemon=True
        )
        self._thread.start()
        logger.info(
            "RemoteControlService listening on %s:%d (token=%s)",
            host,
            port,
            "yes" if self._opts.token else "no",
        )
        return port

    def stop(self) -> None:
        """Wake the selector loop, close all sockets, join the server thread.

        Idempotent.
        """
        if self._stopping.is_set():
            return
        self._stopping.set()
        if self._wake_w is not None:
            try:
                self._wake_w.send(b"x")
            except OSError:
                pass
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        for sock in (self._server_sock, self._wake_r, self._wake_w):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        self._server_sock = None
        self._wake_r = None
        self._wake_w = None
        self._thread = None
        logger.info("RemoteControlService stopped")

    @property
    def port(self) -> int:
        port = self._actual_port
        if port is None:
            raise RuntimeError("RemoteControlService.start() was not called")
        return port

    # ------------------------------------------------------------------
    # Server loop
    # ------------------------------------------------------------------

    def _serve_forever(self) -> None:
        assert self._server_sock is not None
        assert self._wake_r is not None
        sel = selectors.DefaultSelector()
        sel.register(self._server_sock, selectors.EVENT_READ, data=("listener", None))
        sel.register(self._wake_r, selectors.EVENT_READ, data=("wake", None))
        clients: dict[socket.socket, _ClientState] = {}
        token_required = bool(self._opts.token)
        try:
            while not self._stopping.is_set():
                events = sel.select(timeout=0.5)
                for key, _mask in events:
                    kind, _ = key.data
                    if kind == "wake":
                        # Drain the wake pipe; we'll re-check the flag.
                        try:
                            self._wake_r.recv(64)
                        except OSError:
                            pass
                    elif kind == "listener":
                        self._accept_one(sel, clients, token_required)
                    elif kind == "client":
                        sock = key.fileobj
                        assert isinstance(sock, socket.socket)
                        state = clients.get(sock)
                        if state is None:
                            sel.unregister(sock)
                            sock.close()
                            continue
                        self._service_client(sel, clients, sock, state)
        finally:
            for sock in list(clients.keys()):
                try:
                    sel.unregister(sock)
                except (KeyError, ValueError):
                    pass
                try:
                    sock.close()
                except OSError:
                    pass
            try:
                sel.unregister(self._server_sock)
            except (KeyError, ValueError):
                pass
            try:
                sel.unregister(self._wake_r)
            except (KeyError, ValueError):
                pass
            sel.close()

    def _accept_one(
        self,
        sel: selectors.BaseSelector,
        clients: dict[socket.socket, _ClientState],
        token_required: bool,
    ) -> None:
        assert self._server_sock is not None
        try:
            csock, addr = self._server_sock.accept()
        except (BlockingIOError, InterruptedError):
            return
        csock.setblocking(False)
        peer = f"{addr[0]}:{addr[1]}"
        clients[csock] = _ClientState(peer=peer, token_required=token_required)
        sel.register(csock, selectors.EVENT_READ, data=("client", peer))
        logger.info("remote client connected: %s", peer)

    def _service_client(
        self,
        sel: selectors.BaseSelector,
        clients: dict[socket.socket, _ClientState],
        sock: socket.socket,
        state: _ClientState,
    ) -> None:
        try:
            chunk = sock.recv(4096)
        except (BlockingIOError, InterruptedError):
            return
        except OSError as exc:
            logger.info("remote client recv error %s: %s", state.peer, exc)
            self._drop_client(sel, clients, sock, state)
            return
        if not chunk:
            self._drop_client(sel, clients, sock, state)
            return
        state.buffer.extend(chunk)
        # Process every complete line currently in the buffer.
        while True:
            nl = state.buffer.find(LINE_TERMINATOR)
            if nl < 0:
                if len(state.buffer) > MAX_LINE_BYTES:
                    self._reply_error(
                        sock,
                        rid="",
                        code=ErrorCode.INVALID_PARAMS,
                        message=f"request line exceeded {MAX_LINE_BYTES} bytes",
                    )
                    self._drop_client(sel, clients, sock, state)
                return
            line = bytes(state.buffer[:nl])
            del state.buffer[: nl + 1]
            try:
                self._handle_line(sock, state, line)
            except OSError as exc:
                logger.info("remote client write error %s: %s", state.peer, exc)
                self._drop_client(sel, clients, sock, state)
                return

    def _drop_client(
        self,
        sel: selectors.BaseSelector,
        clients: dict[socket.socket, _ClientState],
        sock: socket.socket,
        state: _ClientState,
    ) -> None:
        try:
            sel.unregister(sock)
        except (KeyError, ValueError):
            pass
        try:
            sock.close()
        except OSError:
            pass
        clients.pop(sock, None)
        logger.info("remote client disconnected: %s", state.peer)

    # ------------------------------------------------------------------
    # Per-request handling
    # ------------------------------------------------------------------

    def _handle_line(
        self, sock: socket.socket, state: _ClientState, line: bytes
    ) -> None:
        rid = ""
        try:
            raw = decode_line(line)
            req = parse_request(raw)
            rid = req.id
            if req.method == "auth":
                self._handle_auth(sock, state, req.id, req.params)
                return
            if not state.authed:
                self._reply_error(
                    sock,
                    rid=req.id,
                    code=ErrorCode.UNAUTHORIZED,
                    message="auth required: send {method: 'auth', params: {token: ...}} first",
                )
                return
            spec = METHOD_REGISTRY.get(req.method)
            if spec is None:
                self._reply_error(
                    sock,
                    rid=req.id,
                    code=ErrorCode.UNKNOWN_METHOD,
                    message=f"unknown method: {req.method!r}",
                )
                return
            self._dispatch_on_main(sock, req.id, spec, req.params)
        except RemoteError as exc:
            self._reply_error(sock, rid=rid, code=exc.code, message=exc.message)

    def _handle_auth(
        self,
        sock: socket.socket,
        state: _ClientState,
        rid: str,
        params,
    ) -> None:
        token = _require_str(params, "token")
        configured = self._opts.token
        if not configured:
            self._reply_error(
                sock,
                rid=rid,
                code=ErrorCode.PRECONDITION_FAILED,
                message="auth disabled on this server",
            )
            return
        if hmac.compare_digest(token, configured):
            state.authed = True
            self._reply_ok(sock, rid=rid, result={})
        else:
            self._reply_error(
                sock,
                rid=rid,
                code=ErrorCode.UNAUTHORIZED,
                message="invalid token",
            )

    def _dispatch_on_main(self, sock: socket.socket, rid, spec, params) -> None:
        done = threading.Event()
        holder: dict[str, object] = {}

        def _run() -> None:
            try:
                holder["result"] = spec.handler(self._ctrl, params)
            except RemoteError as exc:
                holder["remote_error"] = exc
            except Exception as exc:  # noqa: BLE001 — Controller error envelope
                logger.exception("handler raised: %s", exc)
                holder["controller_error"] = exc
            finally:
                done.set()

        self._dispatcher.invoke.emit(_run)
        if not done.wait(timeout=spec.timeout_seconds):
            self._reply_error(
                sock,
                rid=rid,
                code=ErrorCode.TIMEOUT,
                message=f"handler did not complete within {spec.timeout_seconds}s",
            )
            return
        if "remote_error" in holder:
            exc = holder["remote_error"]
            assert isinstance(exc, RemoteError)
            self._reply_error(sock, rid=rid, code=exc.code, message=exc.message)
            return
        if "controller_error" in holder:
            err = holder["controller_error"]
            self._reply_error(
                sock,
                rid=rid,
                code=ErrorCode.CONTROLLER_ERROR,
                message=str(err),
            )
            return
        result = holder["result"]
        assert isinstance(result, dict) or hasattr(result, "items")
        self._reply_ok(sock, rid=rid, result=result)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Reply helpers
    # ------------------------------------------------------------------

    def _reply_ok(self, sock: socket.socket, *, rid: str, result) -> None:
        resp = Response(id=rid, ok=True, result=result)
        sock.sendall(encode_line(resp.to_wire()))

    def _reply_error(
        self,
        sock: socket.socket,
        *,
        rid: str,
        code: ErrorCode,
        message: str,
    ) -> None:
        env = ErrorEnvelope(code=code.value, message=message)
        resp = Response(id=rid, ok=False, error=env)
        try:
            sock.sendall(encode_line(resp.to_wire()))
        except OSError:
            pass
