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

from .change_categories import (
    CAT_CFG_EDITED,
    CAT_CONTEXT_CHANGED,
    CAT_DEVICE_CHANGED,
    CAT_SOC_CHANGED,
    CAT_TAB_CHANGED,
    category_for,
)
from .dispatch import METHOD_REGISTRY
from .errors import ErrorCode, ErrorEnvelope, RemoteError
from .events import EVENT_SERIALIZERS, wire_event_name
from .framing import LINE_TERMINATOR, MAX_LINE_BYTES, decode_line, encode_line
from .param_spec import validate_params
from .wire import Response, _require_str, parse_request

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
        "change_buffer",
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
        # Coarse GUI-change tally caused by *others* since this connection last
        # received a summary: (category, object_id) -> count. Drives the
        # per-reply gui_changes notification and the stale-operation guard.
        self.change_buffer: dict[tuple[str, Optional[str]], int] = {}


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
        # The client whose RPC handler is currently executing on the main
        # thread (set around each marshalled handler). Change-buffer bumps skip
        # it so a connection is not "notified" of its own RPC-caused changes.
        self._originating_state: Optional[_ClientState] = None

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
            # Runs on the Qt main thread (EventBus emits synchronously).
            # Tally into every client's change buffer first (independent of
            # whether they subscribed to the push), skipping the originator.
            cat = category_for(event, payload)
            logger.debug(
                "event-flow: bus event=%s -> category=%s (originating=%s)",
                event.value,
                cat,
                self._originating_state.peer if self._originating_state else None,
            )
            if cat is not None:
                self._bump_change_buffer(cat[0], cat[1])
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

    def _bump_change_buffer(self, category: str, object_id: Optional[str]) -> None:
        """Tally a GUI change into every client's buffer, skipping originator.

        Runs on the Qt main thread. The originating connection (the one whose
        RPC handler caused this change) is skipped so it is not notified of /
        guarded against its own action.
        """
        key = (category, object_id)
        with self._clients_lock:
            clients = list(self._clients.values())
        bumped = 0
        skipped_originator = 0
        for state in clients:
            if state is self._originating_state:
                skipped_originator += 1
                continue
            state.change_buffer[key] = state.change_buffer.get(key, 0) + 1
            bumped += 1
        logger.debug(
            "event-flow: bump %s -> %d client(s) tallied, %d skipped (originator), "
            "%d total clients",
            key,
            bumped,
            skipped_originator,
            len(clients),
        )

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
        the id from every client's subscription set.
        """
        # Tally cfg edits into the change buffer (skip originator). editor_closed
        # is a tab/session lifecycle signal -> tab_changed; editor_changed is a
        # cfg edit -> cfg_edited, keyed by editor_id so the stale guard can match
        # a run/commit that depends on that editor.
        if event_name == "editor_changed":
            self._bump_change_buffer(CAT_CFG_EDITED, editor_id)
        elif event_name == "editor_closed":
            self._bump_change_buffer(CAT_TAB_CHANGED, editor_id)
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
            # changes.poll drains this connection's change buffer on demand
            # (state-owning). Like any reply it read-clears via _reply_ok, so
            # "polling" counts as being informed and releases the stale guard.
            if req.method == "changes.poll":
                self._reply_ok(state, rid=req.id, result={})
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
        done = threading.Event()
        holder: dict[str, object] = {}

        def _run() -> None:
            # Runs on the Qt main thread (where the change buffer is written and
            # CfgEditorService lives), so the stale guard reads consistent state.
            # Mark this client as the originator so synchronous EventBus emits
            # inside the handler don't bump its own change buffer.
            self._originating_state = state
            try:
                self._guard_stale(state, method, params)
                holder["result"] = spec.handler(self._ctrl, params)
            except RemoteError as exc:
                holder["remote_error"] = exc
            except Exception as exc:  # noqa: BLE001 — Controller error envelope
                logger.exception("handler raised: %s", exc)
                holder["controller_error"] = exc
            finally:
                self._originating_state = None
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

    def _guard_stale(self, state: _ClientState, method: str, params) -> None:
        """Block a high-risk op if a *dependency* was changed by someone else.

        An inform-once gate: it raises ``PRECONDITION_FAILED`` only while the
        change buffer still holds an un-surfaced change affecting the op's
        dependency. The error reply drains the buffer (``_reply_error`` →
        ``_drain_change_buffer``), so the agent is informed and an immediate
        retry passes — past that point it is the agent's responsibility.

        Runs on the Qt main thread (called from the marshalled handler), so
        reading ``change_buffer`` and resolving tab→editor is consistent.
        """
        if method == "run.start":
            tab_id = params.get("tab_id") if hasattr(params, "get") else None
            if not isinstance(tab_id, str):
                return
            editor_id = self._safe_editor_id_for_owner(tab_id)
            # A run depends on everything that affects its result: the tab's own
            # cfg (keyed by its editor) and existence, plus the global inputs it
            # lowers/runs against — SoC (hardware), active context (cfg evals
            # against md / references ml) and devices (flux bias / sources used
            # by the run). Derived categories (run_changed, predictor_changed)
            # are NOT inputs and are excluded (they'd self-block / false-positive).
            tab_stale = self._buffer_has(
                state, CAT_CFG_EDITED, editor_id
            ) or self._buffer_has(state, CAT_TAB_CHANGED, tab_id)
            global_stale = (
                self._buffer_has_global(state, CAT_SOC_CHANGED)
                or self._buffer_has_global(state, CAT_CONTEXT_CHANGED)
                or self._buffer_has_device(state)
            )
            if tab_stale or global_stale:
                logger.debug(
                    "event-flow: stale guard BLOCK run.start tab=%s editor=%s "
                    "tab_stale=%s global_stale=%s (buffer=%s)",
                    tab_id,
                    editor_id,
                    tab_stale,
                    global_stale,
                    dict(state.change_buffer),
                )
                raise RemoteError(
                    ErrorCode.PRECONDITION_FAILED,
                    f"tab {tab_id!r} or a global resource (SoC / active context) "
                    "was changed in the GUI since you last saw it; review then retry",
                    reason="stale_tab",
                )
        elif method == "editor.commit":
            editor_id = params.get("editor_id") if hasattr(params, "get") else None
            if not isinstance(editor_id, str):
                return
            # commit lowers the draft's EvalValues against the current MetaDict
            # (e.g. freq=r_f -> concrete), so a context/md/ml change shifts what
            # the commit lands — same reasoning as run depending on context.
            # (SoC/device don't affect lowering, so they don't block commit.)
            if self._buffer_has(
                state, CAT_CFG_EDITED, editor_id
            ) or self._buffer_has_global(state, CAT_CONTEXT_CHANGED):
                raise RemoteError(
                    ErrorCode.PRECONDITION_FAILED,
                    f"editor {editor_id!r} or the active context was changed in "
                    "the GUI since you last saw it; review then retry",
                    reason="stale_editor",
                )

    @staticmethod
    def _buffer_has(
        state: _ClientState, category: str, object_id: Optional[str]
    ) -> bool:
        if object_id is None:
            return False
        return state.change_buffer.get((category, object_id), 0) > 0

    @staticmethod
    def _buffer_has_global(state: _ClientState, category: str) -> bool:
        """True if a global (object-less) change of ``category`` is buffered."""
        return state.change_buffer.get((category, None), 0) > 0

    @staticmethod
    def _buffer_has_device(state: _ClientState) -> bool:
        """True if any device change is buffered (keyed by device name or None)."""
        return any(
            cat == CAT_DEVICE_CHANGED and count > 0
            for (cat, _obj), count in state.change_buffer.items()
        )

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

    def _drain_change_buffer(
        self, state: _ClientState
    ) -> Optional[list[dict[str, object]]]:
        """Snapshot + clear a client's change buffer for a reply (read-clears).

        Returns a JSON-safe summary list, or ``None`` when empty. Draining here
        is what counts as the agent "having been informed" — it both surfaces
        the notification and releases the stale guard for those items.
        """
        if not state.change_buffer:
            return None
        summary = [
            {"category": cat, "object_id": obj, "count": count}
            for (cat, obj), count in sorted(
                state.change_buffer.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")
            )
        ]
        logger.debug(
            "event-flow: drain %d change(s) to client %s (read-clears): %s",
            len(summary),
            state.peer,
            summary,
        )
        state.change_buffer.clear()
        return summary

    def _reply_ok(self, state: _ClientState, *, rid: str, result) -> None:
        resp = Response(
            id=rid, ok=True, result=result, gui_changes=self._drain_change_buffer(state)
        )
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
        resp = Response(
            id=rid, ok=False, error=env, gui_changes=self._drain_change_buffer(state)
        )
        try:
            line = encode_line(resp.to_wire())
        except Exception:
            logger.exception("failed to encode error reply for %s", rid)
            return
        self._enqueue(state, line, is_push=False)


# Re-export base64 helpers used by view.screenshot.
__all__ = ["ControlOptions", "RemoteControlService", "base64"]
