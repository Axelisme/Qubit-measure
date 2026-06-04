"""RemoteControlAdapter — local socket interface for automation agents.

Bind defaults to 127.0.0.1. NDJSON request/response (plus optional event push)
over plain TCP. This is the fluxdep-gui's *second View* (driving adapter): the
RPC face onto the ``Controller``, peer to the Qt ``MainWindow``.

Compared to measure-gui's adapter this is deliberately leaner — fluxdep has no
CfgEditor sessions, no diagnostic fan-out channel, and no off-main blocking
handlers — so it offers only:
  - request / reply (validate params on the IO thread, run the handler on the
    Qt main thread, reply);
  - EventBus push (subscribe the 6 fluxdep payload types, serialize and push to
    subscribed clients);
  - token auth + the ``wire.version`` / ``events.*`` handshakes.

Threading:
  - Server thread (daemon): listens, accepts, reads request lines per client,
    parses, marshals handlers onto the Qt main thread via a queued signal,
    waits (timeout-bounded) for the handler, then enqueues the reply onto that
    client's outbound queue.
  - Per-client writer thread (daemon): the sole writer to its client socket;
    drains a ``queue.Queue`` of pre-encoded reply / event lines.
  - Main thread (Qt): runs dispatch handlers and EventBus callbacks (the bus
    callbacks serialize a payload and enqueue bytes onto subscribed clients).
  - Shutdown is initiated from the main thread; it unsubscribes EventBus
    callbacks synchronously, enqueues sentinels to every writer, joins writers,
    then wakes and joins the server thread via a ``socket.socketpair``.
"""

from __future__ import annotations

import hmac
import logging
import queue
import selectors
import socket
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Mapping, Optional

from qtpy.QtCore import QObject, Qt, Signal  # type: ignore[attr-defined]

from zcu_tools.gui.app.fluxdep.event_bus import EventBus, Payload

if TYPE_CHECKING:
    # Type-only: importing Controller at runtime would not cycle here, but the
    # string annotation keeps the import graph lean and pyright happy.
    from zcu_tools.gui.app.fluxdep.controller import Controller

from zcu_tools.gui.remote.errors import ErrorCode, ErrorEnvelope, RemoteError
from zcu_tools.gui.remote.framing import (
    LINE_TERMINATOR,
    MAX_LINE_BYTES,
    decode_line,
    encode_line,
)
from zcu_tools.gui.remote.param_spec import validate_params
from zcu_tools.gui.remote.wire import Response, parse_request, require_str

from .dispatch import METHOD_REGISTRY
from .events import EVENT_SERIALIZERS, wire_event_name
from .wire_version import GUI_VERSION, WIRE_VERSION

logger = logging.getLogger(__name__)


# Outbound queue capacity per client. Slow / wedged readers cause messages to
# be dropped past this point (with a WARN log); see ``_QUEUE_DROP_BUDGET``.
_OUTBOUND_QUEUE_MAX = 256

# Consecutive drops on the same client before we proactively close it.
_QUEUE_DROP_BUDGET = 8

_SHUTDOWN_SENTINEL: bytes = b""


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
    via Qt's queued connection — the supported way to marshal arbitrary work
    onto the Qt event loop from a foreign thread.
    """

    invoke = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.invoke.connect(self._run, type=Qt.ConnectionType.QueuedConnection)  # type: ignore[call-arg]

    def _run(self, fn: Callable[[], None]) -> None:
        fn()


class _ClientState:
    """Tracks per-connection authentication, buffer, subscriptions, writer."""

    __slots__ = (
        "authed",
        "buffer",
        "peer",
        "subscribed",
        "outbound",
        "writer_thread",
        "consecutive_drops",
        "closing",
    )

    def __init__(self, peer: str, token_required: bool) -> None:
        self.peer = peer
        # If no token is configured, every client starts authenticated.
        self.authed = not token_required
        self.buffer = bytearray()
        self.subscribed: set[str] = set()
        self.outbound: queue.Queue[bytes] = queue.Queue(maxsize=_OUTBOUND_QUEUE_MAX)
        self.writer_thread: Optional[threading.Thread] = None
        self.consecutive_drops: int = 0
        self.closing: bool = False


class RemoteControlAdapter:
    """Driving adapter: an NDJSON RPC face onto the fluxdep ``Controller``.

    Holds the ``Controller`` (command face) and pulls ``EventBus`` from it via
    ``controller.bus``. Dispatch handlers receive *this adapter*, so they reach
    commands through ``adapter.ctrl.<m>``. Construct after the Controller exists;
    inert until ``start()``.
    """

    def __init__(self, controller: "Controller", opts: ControlOptions) -> None:
        if opts.allow_external and not opts.token:
            raise RuntimeError(
                "Remote control must specify a token when allow_external is set"
            )
        # Public: dispatch handlers reach the command face through ``adapter.ctrl``.
        self.ctrl = controller
        self._opts = opts
        self._dispatcher = _MainThreadDispatcher()
        self._stopping = threading.Event()
        self._server_sock: Optional[socket.socket] = None
        self._wake_r: Optional[socket.socket] = None
        self._wake_w: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._actual_port: Optional[int] = None
        self._clients: dict[socket.socket, _ClientState] = {}
        self._clients_lock = threading.Lock()
        # EventBus subscriptions registered in start(); unsubscribed in stop().
        self._bus: Optional[EventBus] = None
        self._bus_subs: list[tuple[type[Payload], Callable[[Payload], None]]] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> int:
        """Bind, listen, spawn the server thread, hook EventBus.

        Returns the actual bound port.
        """
        if self._thread is not None:
            raise RuntimeError("RemoteControlAdapter.start() called twice")
        host = self._opts.host()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, self._opts.port))
        except OSError as exc:
            sock.close()
            raise RuntimeError(
                f"RemoteControlAdapter bind {host}:{self._opts.port} failed: {exc}"
            ) from exc
        sock.listen(8)
        sock.setblocking(False)
        self._server_sock = sock
        port = int(sock.getsockname()[1])
        self._actual_port = port

        wake_r, wake_w = socket.socketpair()
        wake_r.setblocking(False)
        self._wake_r, self._wake_w = wake_r, wake_w

        self._subscribe_event_bus()

        self._thread = threading.Thread(
            target=self._serve_forever, name="FluxDepRemoteServer", daemon=True
        )
        self._thread.start()
        logger.info(
            "RemoteControlAdapter listening on %s:%d (token=%s)",
            host,
            port,
            "yes" if self._opts.token else "no",
        )
        return port

    def stop(self) -> None:
        """Wake the selector loop, close all sockets, join threads.

        Idempotent. Must be called from the Qt main thread (it unsubscribes
        EventBus callbacks synchronously).
        """
        if self._stopping.is_set():
            return
        self._stopping.set()

        self._unsubscribe_event_bus()

        with self._clients_lock:
            client_snapshot = list(self._clients.items())
        for _sock, state in client_snapshot:
            state.closing = True
            try:
                state.outbound.put_nowait(_SHUTDOWN_SENTINEL)
            except queue.Full:
                try:
                    state.outbound.get_nowait()
                except queue.Empty:
                    pass
                try:
                    state.outbound.put_nowait(_SHUTDOWN_SENTINEL)
                except queue.Full:
                    pass
        for _sock, state in client_snapshot:
            t = state.writer_thread
            if t is not None and t.is_alive():
                t.join(timeout=2.0)

        if self._wake_w is not None:
            try:
                self._wake_w.send(b"x")
            except OSError:
                pass
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

        with self._clients_lock:
            for sock in list(self._clients.keys()):
                try:
                    sock.close()
                except OSError:
                    pass
            self._clients.clear()
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
        logger.info("RemoteControlAdapter stopped")

    @property
    def port(self) -> int:
        port = self._actual_port
        if port is None:
            raise RuntimeError("RemoteControlAdapter.start() was not called")
        return port

    # ------------------------------------------------------------------
    # EventBus integration
    # ------------------------------------------------------------------

    def _subscribe_event_bus(self) -> None:
        """Subscribe one callback per serialised payload type on the main thread."""
        bus = self.ctrl.bus
        self._bus = bus
        for payload_type in EVENT_SERIALIZERS:
            cb = self._make_bus_callback(payload_type)
            bus.subscribe(payload_type, cb)
            self._bus_subs.append((payload_type, cb))
        logger.debug(
            "event-flow: subscribed %d EventBus payload types: %s",
            len(self._bus_subs),
            [wire_event_name(t) for t, _ in self._bus_subs],
        )

    def _unsubscribe_event_bus(self) -> None:
        if self._bus is None:
            return
        for payload_type, cb in self._bus_subs:
            self._bus.unsubscribe(payload_type, cb)
        self._bus_subs.clear()
        self._bus = None

    def _make_bus_callback(
        self, payload_type: type[Payload]
    ) -> Callable[[Payload], None]:
        serializer = EVENT_SERIALIZERS[payload_type]
        wire_name = wire_event_name(payload_type)

        def _on_event(payload: Payload) -> None:
            # Runs on the Qt main thread. Serialize and push to subscribers.
            try:
                wire_payload = serializer(payload)
            except Exception:  # pragma: no cover — serializer must not raise
                logger.exception("Event serializer for %s raised", wire_name)
                return
            if wire_payload is None:
                return
            try:
                line = encode_line({"event": wire_name, "payload": wire_payload})
            except Exception:
                logger.exception("Failed to encode push line for %s", wire_name)
                return
            self._broadcast(wire_name, line)

        return _on_event

    def _broadcast(self, wire_name: str, line: bytes) -> None:
        with self._clients_lock:
            clients = list(self._clients.values())
        for state in clients:
            if wire_name not in state.subscribed:
                continue
            self._enqueue(state, line, is_push=True)

    def _enqueue(self, state: _ClientState, line: bytes, *, is_push: bool) -> None:
        if state.closing:
            return
        try:
            state.outbound.put_nowait(line)
            state.consecutive_drops = 0
        except queue.Full:
            state.consecutive_drops += 1
            logger.warning(
                "remote client %s outbound queue full (drops=%d, push=%s)",
                state.peer,
                state.consecutive_drops,
                is_push,
            )
            if state.consecutive_drops >= _QUEUE_DROP_BUDGET:
                logger.warning(
                    "remote client %s exceeded drop budget; closing", state.peer
                )
                state.closing = True
                try:
                    state.outbound.get_nowait()
                except queue.Empty:
                    pass
                try:
                    state.outbound.put_nowait(_SHUTDOWN_SENTINEL)
                except queue.Full:
                    pass

    # ------------------------------------------------------------------
    # Server loop
    # ------------------------------------------------------------------

    def _serve_forever(self) -> None:
        assert self._server_sock is not None
        assert self._wake_r is not None
        sel = selectors.DefaultSelector()
        sel.register(self._server_sock, selectors.EVENT_READ, data=("listener", None))
        sel.register(self._wake_r, selectors.EVENT_READ, data=("wake", None))
        token_required = bool(self._opts.token)
        try:
            while not self._stopping.is_set():
                events = sel.select(timeout=0.5)
                for key, _mask in events:
                    kind, _ = key.data
                    if kind == "wake":
                        try:
                            self._wake_r.recv(64)
                        except OSError:
                            pass
                    elif kind == "listener":
                        self._accept_one(sel, token_required)
                    elif kind == "client":
                        sock = key.fileobj
                        assert isinstance(sock, socket.socket)
                        with self._clients_lock:
                            state = self._clients.get(sock)
                        if state is None:
                            sel.unregister(sock)
                            sock.close()
                            continue
                        self._service_client(sel, sock, state)
        finally:
            with self._clients_lock:
                client_socks = list(self._clients.keys())
            for sock in client_socks:
                try:
                    sel.unregister(sock)
                except (KeyError, ValueError):
                    pass
            for extra in (self._server_sock, self._wake_r):
                try:
                    sel.unregister(extra)
                except (KeyError, ValueError):
                    pass
            sel.close()

    def _accept_one(self, sel: selectors.BaseSelector, token_required: bool) -> None:
        assert self._server_sock is not None
        try:
            csock, addr = self._server_sock.accept()
        except (BlockingIOError, InterruptedError):
            return
        csock.setblocking(False)
        peer = f"{addr[0]}:{addr[1]}"
        state = _ClientState(peer=peer, token_required=token_required)
        with self._clients_lock:
            self._clients[csock] = state
        sel.register(csock, selectors.EVENT_READ, data=("client", peer))
        writer = threading.Thread(
            target=self._client_writer,
            args=(csock, state),
            name=f"FluxDepRemoteWriter[{peer}]",
            daemon=True,
        )
        state.writer_thread = writer
        writer.start()
        logger.info("remote client connected: %s", peer)

    def _client_writer(self, sock: socket.socket, state: _ClientState) -> None:
        """Drain the outbound queue to the socket; exits on sentinel / close."""
        while True:
            try:
                line = state.outbound.get(timeout=1.0)
            except queue.Empty:
                if state.closing or self._stopping.is_set():
                    return
                continue
            if line is _SHUTDOWN_SENTINEL or line == _SHUTDOWN_SENTINEL:
                return
            try:
                sock.sendall(line)
            except OSError as exc:
                logger.info("remote client writer %s exit on send: %s", state.peer, exc)
                state.closing = True
                return

    def _service_client(
        self, sel: selectors.BaseSelector, sock: socket.socket, state: _ClientState
    ) -> None:
        try:
            chunk = sock.recv(4096)
        except (BlockingIOError, InterruptedError):
            return
        except OSError as exc:
            logger.info("remote client recv error %s: %s", state.peer, exc)
            self._drop_client(sel, sock, state)
            return
        if not chunk:
            self._drop_client(sel, sock, state)
            return
        state.buffer.extend(chunk)
        while True:
            nl = state.buffer.find(LINE_TERMINATOR)
            if nl < 0:
                if len(state.buffer) > MAX_LINE_BYTES:
                    self._reply_error(
                        state,
                        rid="",
                        code=ErrorCode.INVALID_PARAMS,
                        message=f"request line exceeded {MAX_LINE_BYTES} bytes",
                    )
                    self._drop_client(sel, sock, state)
                return
            line = bytes(state.buffer[:nl])
            del state.buffer[: nl + 1]
            self._handle_line(state, line)

    def _drop_client(
        self, sel: selectors.BaseSelector, sock: socket.socket, state: _ClientState
    ) -> None:
        try:
            sel.unregister(sock)
        except (KeyError, ValueError):
            pass
        state.closing = True
        try:
            state.outbound.put_nowait(_SHUTDOWN_SENTINEL)
        except queue.Full:
            pass
        try:
            sock.close()
        except OSError:
            pass
        with self._clients_lock:
            self._clients.pop(sock, None)
        logger.info("remote client disconnected: %s", state.peer)

    # ------------------------------------------------------------------
    # Per-request handling
    # ------------------------------------------------------------------

    def _handle_line(self, state: _ClientState, line: bytes) -> None:
        rid = ""
        try:
            raw = decode_line(line)
            req = parse_request(raw)
            rid = req.id
            # wire.version is a no-auth handshake probe.
            if req.method == "wire.version":
                self._reply_ok(
                    state,
                    rid=req.id,
                    result={"wire_version": WIRE_VERSION, "gui_version": GUI_VERSION},
                )
                return
            if req.method == "auth":
                self._handle_auth(state, req.id, req.params)
                return
            if not state.authed:
                self._reply_error(
                    state,
                    rid=req.id,
                    code=ErrorCode.UNAUTHORIZED,
                    message="auth required: send {method: 'auth', params: {token: ...}} first",
                )
                return
            if req.method == "events.subscribe":
                self._handle_subscribe(state, req.id, req.params)
                return
            if req.method == "events.unsubscribe":
                self._handle_unsubscribe(state, req.id, req.params)
                return
            if req.method == "events.list":
                self._reply_ok(
                    state,
                    rid=req.id,
                    result={
                        "events": sorted(wire_event_name(t) for t in EVENT_SERIALIZERS),
                        "subscribed": sorted(state.subscribed),
                    },
                )
                return
            spec = METHOD_REGISTRY.get(req.method)
            if spec is None:
                self._reply_error(
                    state,
                    rid=req.id,
                    code=ErrorCode.UNKNOWN_METHOD,
                    message=f"unknown method: {req.method!r}",
                )
                return
            # Validate params on the IO thread (pure, no Qt) so malformed
            # requests fail fast without a main-thread hop.
            if spec.params:
                handler_params = validate_params(spec.params, req.params)
            else:
                handler_params = req.params
            self._dispatch_on_main(state, req.id, req.method, spec, handler_params)
        except RemoteError as exc:
            self._reply_error(
                state,
                rid=rid,
                code=exc.code,
                message=exc.message,
                reason=exc.reason,
                data=exc.data,
            )

    def _handle_auth(self, state: _ClientState, rid: str, params) -> None:
        token = require_str(params, "token")
        configured = self._opts.token
        if not configured:
            self._reply_error(
                state,
                rid=rid,
                code=ErrorCode.PRECONDITION_FAILED,
                message="auth disabled on this server",
            )
            return
        if hmac.compare_digest(token, configured):
            state.authed = True
            self._reply_ok(state, rid=rid, result={})
        else:
            self._reply_error(
                state, rid=rid, code=ErrorCode.UNAUTHORIZED, message="invalid token"
            )

    def _handle_subscribe(self, state: _ClientState, rid: str, params) -> None:
        events = params.get("events")
        if not isinstance(events, list):
            raise RemoteError(
                ErrorCode.INVALID_PARAMS, "'events' must be a list of event names"
            )
        whitelist = {wire_event_name(t) for t in EVENT_SERIALIZERS}
        for ev in events:
            if not isinstance(ev, str):
                raise RemoteError(
                    ErrorCode.INVALID_PARAMS,
                    f"event name must be a string, got {type(ev).__name__}",
                )
            if ev not in whitelist:
                raise RemoteError(
                    ErrorCode.INVALID_PARAMS, f"unknown event name: {ev!r}"
                )
        for ev in events:
            assert isinstance(ev, str)
            state.subscribed.add(ev)
        self._reply_ok(state, rid=rid, result={"subscribed": sorted(state.subscribed)})

    def _handle_unsubscribe(self, state: _ClientState, rid: str, params) -> None:
        events = params.get("events")
        if not isinstance(events, list):
            raise RemoteError(
                ErrorCode.INVALID_PARAMS, "'events' must be a list of event names"
            )
        for ev in events:
            if isinstance(ev, str):
                state.subscribed.discard(ev)
        self._reply_ok(state, rid=rid, result={"subscribed": sorted(state.subscribed)})

    def _dispatch_on_main(self, state: _ClientState, rid, method, spec, params) -> None:
        holder: dict[str, object] = {}
        done = threading.Event()

        def _run() -> None:
            # Runs on the Qt main thread (where the State + VersionTable live).
            try:
                holder["result"] = spec.handler(self, params)
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
                state,
                rid=rid,
                code=ErrorCode.TIMEOUT,
                message=f"handler did not complete within {spec.timeout_seconds}s",
            )
            return
        if "remote_error" in holder:
            exc = holder["remote_error"]
            assert isinstance(exc, RemoteError)
            self._reply_error(
                state,
                rid=rid,
                code=exc.code,
                message=exc.message,
                reason=exc.reason,
                data=exc.data,
            )
            return
        if "controller_error" in holder:
            err = holder["controller_error"]
            self._reply_error(
                state, rid=rid, code=ErrorCode.CONTROLLER_ERROR, message=str(err)
            )
            return
        result = holder["result"]
        assert isinstance(result, dict), f"handler {method!r} returned non-dict result"
        self._reply_ok(state, rid=rid, result=result)

    # ------------------------------------------------------------------
    # Reply helpers (enqueue onto outbound queue; never write directly)
    # ------------------------------------------------------------------

    def _reply_ok(self, state: _ClientState, *, rid: str, result) -> None:
        resp = Response(id=rid, ok=True, result=result)
        try:
            line = encode_line(resp.to_wire())
        except Exception:
            logger.exception("failed to encode reply for %s", rid)
            return
        self._enqueue(state, line, is_push=False)

    def _reply_error(
        self,
        state: _ClientState,
        *,
        rid: str,
        code: ErrorCode,
        message: str,
        reason: str = "",
        data: "Optional[dict]" = None,
    ) -> None:
        env = ErrorEnvelope(code=code.value, message=message, reason=reason, data=data)
        resp = Response(id=rid, ok=False, error=env)
        try:
            line = encode_line(resp.to_wire())
        except Exception:
            logger.exception("failed to encode error reply for %s", rid)
            return
        self._enqueue(state, line, is_push=False)


__all__ = ["ControlOptions", "RemoteControlAdapter"]
