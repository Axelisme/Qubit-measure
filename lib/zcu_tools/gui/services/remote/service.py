"""RemoteControlService — local socket interface for automation agents.

Bind defaults to 127.0.0.1. NDJSON request/response (plus optional event push)
over plain TCP.

Threading:
  - Server thread (daemon): listens, accepts, reads request lines per client,
    parses, dispatches handlers via a queued Qt signal onto the main thread,
    waits (timeout-bounded) for the handler to set its event, then enqueues
    the reply onto that client's outbound queue.
  - Per-client writer thread (daemon, one per accepted connection): is the
    sole writer to its client socket. Loops on a ``queue.Queue``; receives
    pre-encoded reply lines and pushed event lines in enqueue order.
  - Main thread (Qt): runs dispatch handlers and EventBus callbacks. The
    bus callbacks serialize a payload and enqueue bytes onto each subscribed
    client's outbound queue.
  - Shutdown is initiated from the main thread; it unsubscribes EventBus
    handlers synchronously on the main thread (Phase 81), enqueues sentinels
    to every writer queue, joins writers, then wakes and joins the server
    thread via a ``socket.socketpair`` self-pipe.
"""

from __future__ import annotations

import base64
import hmac
import logging
import queue
import selectors
import socket
import threading
from dataclasses import dataclass
from typing import Callable, Optional

from qtpy.QtCore import QObject, Qt, Signal  # type: ignore[attr-defined]

from zcu_tools.gui.event_bus import EventBus, GuiEvent, Payload

from .dispatch import METHOD_REGISTRY
from .errors import ErrorCode, ErrorEnvelope, RemoteError
from .events import EVENT_SERIALIZERS, wire_event_name
from .framing import LINE_TERMINATOR, MAX_LINE_BYTES, decode_line, encode_line
from .param_spec import validate_params
from .wire import WIRE_VERSION, Response, _require_str, parse_request

logger = logging.getLogger(__name__)


# Outbound queue capacity per client. Slow / wedged readers cause messages to
# be dropped past this point (with a WARN log); see ``_QUEUE_DROP_BUDGET``.
_OUTBOUND_QUEUE_MAX = 256

# Consecutive drops on the same client before we proactively close it. This
# stops a wedged reader from indefinitely accumulating dropped event pushes.
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
        "editor_ids",
        "subscribed_editors",
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
        # CfgEditor session ids opened by this connection; reclaimed on drop.
        self.editor_ids: set[str] = set()
        # CfgEditor session ids this connection subscribed to for change push.
        self.subscribed_editors: set[str] = set()


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class RemoteControlService:
    """Run an NDJSON dispatcher on a local TCP port.

    Construct after ``Controller`` / ``MainWindow`` exist; the service holds a
    weak relationship to the Controller and pulls EventBus from it via
    ``controller.get_bus()``. Inert until ``start()``.
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
        # Clients held both by the server thread (mutating set) and by the
        # main-thread EventBus callback (iterating set). Guarded by lock.
        self._clients: dict[socket.socket, _ClientState] = {}
        self._clients_lock = threading.Lock()
        # EventBus subscriptions registered in start(); unsubscribed in stop().
        self._bus: Optional[EventBus] = None
        self._bus_subs: list[tuple[GuiEvent, Callable[[Payload], None]]] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> int:
        """Bind, listen, spawn the server thread, hook EventBus.

        Returns the actual bound port.
        """
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

        self._subscribe_event_bus()
        self._wire_editor_change_listener()

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
        """Wake the selector loop, close all sockets, join threads.

        Idempotent. Must be called from the Qt main thread (it unsubscribes
        EventBus callbacks synchronously).
        """
        if self._stopping.is_set():
            return
        self._stopping.set()

        # 1) Unsubscribe EventBus + editor change stream so no more enqueues.
        self._unsubscribe_event_bus()
        self._unwire_editor_change_listener()

        # 2) Signal every writer to exit; join them.
        with self._clients_lock:
            client_snapshot = list(self._clients.items())
        for _sock, state in client_snapshot:
            # stop() runs on the main thread, so reclaim editors directly.
            self._reclaim_editors(state, marshal=False)
            state.closing = True
            try:
                state.outbound.put_nowait(_SHUTDOWN_SENTINEL)
            except queue.Full:
                # Force-drain one slot then push sentinel so writer wakes.
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

        # 3) Wake the server selector loop.
        if self._wake_w is not None:
            try:
                self._wake_w.send(b"x")
            except OSError:
                pass
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

        # 4) Close all sockets.
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
        logger.info("RemoteControlService stopped")

    @property
    def port(self) -> int:
        port = self._actual_port
        if port is None:
            raise RuntimeError("RemoteControlService.start() was not called")
        return port

    # ------------------------------------------------------------------
    # EventBus integration
    # ------------------------------------------------------------------

    def _subscribe_event_bus(self) -> None:
        """Subscribe one callback per serialised GuiEvent on the main thread."""
        get_bus = getattr(self._ctrl, "get_bus", None)
        if get_bus is None:
            logger.warning("Controller has no get_bus(); event push disabled")
            return
        bus = get_bus()
        if not isinstance(bus, EventBus):
            logger.warning(
                "Controller.get_bus() returned non-EventBus; event push disabled"
            )
            return
        self._bus = bus
        for event in EVENT_SERIALIZERS.keys():
            cb = self._make_bus_callback(event)
            try:
                bus.subscribe(event, cb)
            except Exception:  # pragma: no cover — bus.subscribe is straightforward
                logger.exception("Failed to subscribe %s on EventBus", event)
                continue
            self._bus_subs.append((event, cb))
        logger.debug(
            "event-flow: subscribed %d EventBus events for push+buffer: %s",
            len(self._bus_subs),
            [e.value for e, _ in self._bus_subs],
        )

    def _unsubscribe_event_bus(self) -> None:
        if self._bus is None:
            return
        for event, cb in self._bus_subs:
            try:
                self._bus.unsubscribe(event, cb)
            except Exception:  # pragma: no cover
                logger.exception("Failed to unsubscribe %s on EventBus", event)
        self._bus_subs.clear()
        self._bus = None

    def _make_bus_callback(self, event: GuiEvent) -> Callable[[Payload], None]:
        serializer = EVENT_SERIALIZERS[event]
        wire_name = wire_event_name(event)

        def _on_event(payload: Payload) -> None:
            # Runs on the Qt main thread. Serialize and push to subscribers; this
            # is the agent's notification face ("what changed"). Resource version
            # bookkeeping happens at the mutation site (State.version), not here.
            try:
                wire_payload = serializer(payload)
            except Exception:  # pragma: no cover — serializer must not raise
                logger.exception("Event serializer for %s raised", event)
                return
            if wire_payload is None:
                return
            try:
                line = encode_line({"event": wire_name, "payload": wire_payload})
            except Exception:
                logger.exception(
                    "Failed to encode push line for %s payload=%r", event, wire_payload
                )
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

    # ------------------------------------------------------------------
    # CfgEditor per-session change stream (independent of EventBus)
    # ------------------------------------------------------------------

    def _wire_editor_change_listener(self) -> None:
        """Inject ``_on_editor_event`` into the CfgEditorService (via ctrl)."""
        setter = getattr(self._ctrl, "set_cfg_editor_change_listener", None)
        if setter is None:
            logger.warning(
                "Controller has no set_cfg_editor_change_listener(); "
                "editor change push disabled"
            )
            return
        setter(self._on_editor_event)

    def _unwire_editor_change_listener(self) -> None:
        setter = getattr(self._ctrl, "set_cfg_editor_change_listener", None)
        if setter is not None:
            setter(None)

    def _on_editor_event(self, editor_id: str, event_name: str, payload: dict) -> None:
        """Push a per-editor notification. Runs on the Qt main thread.

        ``event_name`` ∈ {editor_changed, editor_closed}. Only clients that
        subscribed to ``editor_id`` receive it. On editor_closed we also drop
        the id from every client's subscription set. (The editor's resource
        version is bumped at the edit site, not here.)
        """
        body = dict(payload)
        body["editor_id"] = editor_id
        try:
            line = encode_line({"event": event_name, "payload": body})
        except Exception:
            logger.exception(
                "failed to encode editor push %s/%s", editor_id, event_name
            )
            return
        with self._clients_lock:
            clients = list(self._clients.values())
        for state in clients:
            if editor_id not in state.subscribed_editors:
                continue
            self._enqueue(state, line, is_push=True)
            if event_name == "editor_closed":
                state.subscribed_editors.discard(editor_id)

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
                    "remote client %s exceeded drop budget; closing",
                    state.peer,
                )
                state.closing = True
                # The writer thread will exit when it next finds the queue
                # empty after we forcefully push a sentinel; do so on a
                # best-effort basis after draining one slot.
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
        token_required: bool,
    ) -> None:
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
        # Spawn writer thread (sole writer for this socket).
        writer = threading.Thread(
            target=self._client_writer,
            args=(csock, state),
            name=f"RemoteControlWriter[{peer}]",
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
        self,
        sel: selectors.BaseSelector,
        sock: socket.socket,
        state: _ClientState,
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
        # Process every complete line currently in the buffer.
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
        self,
        sel: selectors.BaseSelector,
        sock: socket.socket,
        state: _ClientState,
    ) -> None:
        try:
            sel.unregister(sock)
        except (KeyError, ValueError):
            pass
        state.closing = True
        # Wake writer to exit.
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
        # Reclaim this connection's CfgEditor sessions on the main thread
        # (LiveModel teardown is main-thread-owned). Fire-and-forget: the
        # client is already gone, so no reply is awaited.
        self._reclaim_editors(state, marshal=True)
        logger.info("remote client disconnected: %s", state.peer)

    def _reclaim_editors(self, state: _ClientState, *, marshal: bool) -> None:
        """Discard CfgEditor sessions opened by ``state``; clears its id set.

        ``marshal=True`` schedules the discard on the Qt main thread (use from
        the IO/server thread, e.g. ``_drop_client``); ``marshal=False`` calls
        directly (use from the main thread, e.g. ``stop``).
        """
        ids = list(state.editor_ids)
        state.editor_ids.clear()
        if not ids:
            return
        discard = getattr(self._ctrl, "discard_cfg_editors", None)
        if discard is None:
            return

        def _run() -> None:
            try:
                discard(ids)
            except Exception:  # pragma: no cover — best-effort cleanup
                logger.exception("failed to reclaim editor sessions %r", ids)

        if marshal:
            self._dispatcher.invoke.emit(_run)
        else:
            _run()

    # ------------------------------------------------------------------
    # Per-request handling
    # ------------------------------------------------------------------

    def _handle_line(self, state: _ClientState, line: bytes) -> None:
        rid = ""
        try:
            raw = decode_line(line)
            req = parse_request(raw)
            rid = req.id
            # wire.version is a no-auth handshake probe: it lets a caller read
            # the server's wire-protocol version before authenticating, so a
            # stale GUI process is detectable on connect.
            if req.method == "wire.version":
                self._reply_ok(
                    state, rid=req.id, result={"wire_version": WIRE_VERSION}
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
            # Subscription methods are handled by the service itself (state-owning).
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
                        "events": sorted(wire_event_name(e) for e in EVENT_SERIALIZERS),
                        "subscribed": sorted(state.subscribed),
                    },
                )
                return
            # editor.subscribe/unsubscribe are state-owning (per-connection
            # editor subscription set), so handled here, not via dispatch.
            if req.method == "editor.subscribe":
                self._handle_editor_subscribe(state, req.id, req.params, subscribe=True)
                return
            if req.method == "editor.unsubscribe":
                self._handle_editor_subscribe(
                    state, req.id, req.params, subscribe=False
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
            # Validate params against the method's ParamSpec contract on the IO
            # thread (pure, no Qt needed) so malformed requests fail fast without
            # consuming a main-thread hop. Methods that have not declared params
            # yet pass their raw params through unchanged.
            if spec.params:
                handler_params = validate_params(spec.params, req.params)
            else:
                handler_params = req.params
            self._dispatch_on_main(state, req.id, req.method, spec, handler_params)
        except RemoteError as exc:
            self._reply_error(
                state, rid=rid, code=exc.code, message=exc.message, reason=exc.reason
            )

    def _handle_auth(
        self,
        state: _ClientState,
        rid: str,
        params,
    ) -> None:
        token = _require_str(params, "token")
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
                state,
                rid=rid,
                code=ErrorCode.UNAUTHORIZED,
                message="invalid token",
            )

    def _handle_editor_subscribe(
        self, state: _ClientState, rid: str, params, *, subscribe: bool
    ) -> None:
        editor_id = params.get("editor_id")
        if not isinstance(editor_id, str) or not editor_id:
            raise RemoteError(
                ErrorCode.INVALID_PARAMS, "'editor_id' must be a non-empty string"
            )
        # No existence check: subscription is a pure per-connection filter. A
        # client may subscribe before/around open; pushes only flow for live
        # sessions, and editor_closed cleans the set.
        if subscribe:
            state.subscribed_editors.add(editor_id)
        else:
            state.subscribed_editors.discard(editor_id)
        self._reply_ok(
            state,
            rid=rid,
            result={"subscribed_editors": sorted(state.subscribed_editors)},
        )

    def _handle_subscribe(self, state: _ClientState, rid: str, params) -> None:
        events = params.get("events")
        if not isinstance(events, list):
            raise RemoteError(
                ErrorCode.INVALID_PARAMS, "'events' must be a list of event names"
            )
        whitelist = {wire_event_name(e) for e in EVENT_SERIALIZERS}
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

        if spec.off_main_thread:
            # Blocking handler (e.g. device.wait_setup): run on THIS IO worker
            # thread, never the main thread — marshalling it onto the main
            # thread would deadlock (it would occupy the event loop that must
            # dispatch the worker signal it awaits). It must only do thread-safe
            # waiting and must not touch the change buffer / stale guard /
            # origin scope, so none of those are set here.
            try:
                holder["result"] = spec.handler(self._ctrl, params)
            except RemoteError as exc:
                holder["remote_error"] = exc
            except Exception as exc:  # noqa: BLE001 — Controller error envelope
                logger.exception("off-main handler raised: %s", exc)
                holder["controller_error"] = exc
        else:
            done = threading.Event()

            def _run() -> None:
                # Runs on the Qt main thread (where CfgEditorService and the
                # version table live), so the version guard's compare-and-act is
                # atomic against any other GUI write.
                try:
                    self._guard_versions(params)
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
                state, rid=rid, code=exc.code, message=exc.message, reason=exc.reason
            )
            return
        if "controller_error" in holder:
            err = holder["controller_error"]
            self._reply_error(
                state,
                rid=rid,
                code=ErrorCode.CONTROLLER_ERROR,
                message=str(err),
            )
            return
        result = holder["result"]
        assert isinstance(result, dict) or hasattr(result, "items")
        self._track_editor_lifecycle(state, method, params, result)
        self._reply_ok(state, rid=rid, result=result)  # type: ignore[arg-type]

    def _guard_versions(self, params) -> None:
        """Atomically reject an op whose declared resource versions are stale.

        Optimistic concurrency, If-Match style: the mcp layer (which owns the
        dependency policy) attaches an ``expected_versions`` dict mapping the
        resource keys this op depends on to the versions it last saw. The server
        is pure mechanism — it compares each given key against the current
        ``VersionTable`` and does not care *why* those keys matter. A mismatch
        (someone, possibly a human, mutated a dependency since the caller read
        it; or the resource was dropped and now reads 0) raises
        ``PRECONDITION_FAILED`` carrying the current versions of the offending
        keys so the caller can resync and retry.

        Absent/empty ``expected_versions`` means no check (same as a plain RPC).

        Runs on the Qt main thread inside the handler's synchronous ``_run()``
        sequence, so the compare-and-act is atomic against any other GUI write
        (real-user actions also marshal onto the main thread) — no TOCTOU.
        """
        expected = params.get("expected_versions") if hasattr(params, "get") else None
        if not expected:
            return
        if not isinstance(expected, dict):
            raise RemoteError(
                ErrorCode.INVALID_PARAMS,
                "'expected_versions' must be an object of resource->version",
            )
        current = self._ctrl_resource_versions()
        mismatched = {
            key: current.get(key, 0)
            for key, want in expected.items()
            if current.get(key, 0) != want
        }
        if mismatched:
            # The caller (mcp) resyncs by re-reading resources.versions on this
            # error — pure read-via-snapshot, so the error carries no version
            # numbers itself (they stay RPC<->mcp bookkeeping, never on the
            # agent-facing message).
            logger.debug(
                "version guard BLOCK: expected=%s mismatched(current)=%s",
                expected,
                mismatched,
            )
            raise RemoteError(
                ErrorCode.PRECONDITION_FAILED,
                "a resource you depend on was changed in the GUI since you last "
                "saw it; review then retry",
                reason="stale_version",
            )

    def _ctrl_resource_versions(self) -> dict[str, int]:
        getter = getattr(self._ctrl, "resources_versions", None)
        if getter is None:
            return {}
        return dict(getter())

    def _safe_editor_id_for_owner(self, owner_key: str) -> Optional[str]:
        getter = getattr(self._ctrl, "editor_id_for_owner", None)
        if getter is None:
            return None
        try:
            return getter(owner_key)
        except Exception:  # pragma: no cover — defensive
            return None

    def _track_editor_lifecycle(
        self, state: _ClientState, method: str, params, result
    ) -> None:
        """Record/forget CfgEditor session ids per connection.

        ``editor.open`` binds the returned id to this client so a disconnect
        reclaims it; ``commit``/``discard`` forget it (the session is already
        gone server-side). Runs on the IO thread, where ``state.editor_ids``
        lives.
        """
        if method == "editor.open":
            editor_id = result.get("editor_id") if isinstance(result, dict) else None
            if isinstance(editor_id, str):
                state.editor_ids.add(editor_id)
        elif method in ("editor.commit", "editor.discard"):
            editor_id = params.get("editor_id") if hasattr(params, "get") else None
            if isinstance(editor_id, str):
                state.editor_ids.discard(editor_id)

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
    ) -> None:
        env = ErrorEnvelope(code=code.value, message=message, reason=reason)
        resp = Response(id=rid, ok=False, error=env)
        try:
            line = encode_line(resp.to_wire())
        except Exception:
            logger.exception("failed to encode error reply for %s", rid)
            return
        self._enqueue(state, line, is_push=False)


# Re-export base64 helpers used by view.screenshot.
__all__ = ["ControlOptions", "RemoteControlService", "base64"]
