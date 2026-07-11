"""NdjsonRpcEndpoint — the shared transport for the GUI apps' remote control.

This is the *pure transport* half of every app's ``RemoteControlAdapter``. It
owns only how bytes enter and leave the socket, never what a request *means*:

  - the server socket lifecycle (bind / listen / selector loop / socketpair-wake
    / server thread, plus the start/stop thread-join orchestration);
  - per-connection framing + delivery: each accepted client gets a
    :class:`ClientLink` (recv buffer + NDJSON line framing + an outbound queue
    drained by a sole-writer thread, with drop-budget backpressure);
  - request parsing (``decode_line`` + ``parse_request``) and the built-in
    ``wire.version`` / ``auth`` handshakes (the only two methods the endpoint
    understands — both need only injected constants + the configured token);
  - reply encoding (:meth:`reply_ok` / :meth:`reply_error`) and lock-safe eager
    / lazy push fan-out primitives (:meth:`broadcast` / :meth:`broadcast_lazy`);
  - the :class:`MainThreadDispatcher` Qt-main-thread marshal primitive (a reusable
    QObject each app composes into its own ``_dispatch_on_main``).

Everything past the handshake is the app's: the endpoint hands each parsed,
authenticated :class:`Request` to the injected :class:`EndpointRouter` and stops
there. The router decides which methods exist, how/where their handlers run
(main-thread marshal, version guard, off-main blocking), and how events are
serialized and pushed. The endpoint is zero-knowledge of all of it.

Threading:
  - Server thread (daemon): selector loop — accept, recv, frame; built-in
    handshakes reply on this IO thread; everything else is handed to
    ``router.route`` (still on the IO thread — the router owns any main-thread
    marshal).
  - Per-client writer thread (daemon): the sole writer to its client socket;
    drains a ``queue.Queue`` of pre-encoded reply / push lines.
  - Main thread (Qt): the router's handlers + the app's event-bus callbacks run
    here; pushes originate here and call :meth:`broadcast`.
  - Shutdown is initiated from the main thread: ``stop()`` calls the router's
    ``on_client_close`` per client, enqueues sentinels to every writer, joins
    writers, then wakes and joins the server thread via a ``socket.socketpair``.
"""

from __future__ import annotations

import hmac
import logging
import queue
import selectors
import socket
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar

from qtpy.QtCore import QObject, Qt, Signal  # type: ignore[attr-defined]

from zcu_tools.gui.remote.errors import ErrorCode, ErrorEnvelope, RemoteError
from zcu_tools.gui.remote.framing import (
    LINE_TERMINATOR,
    MAX_LINE_BYTES,
    decode_line,
    encode_line,
)
from zcu_tools.gui.remote.wire import Response, parse_request, require_str

logger = logging.getLogger(__name__)


# Outbound queue capacity per client. Slow / wedged readers cause messages to
# be dropped past this point (with a WARN log); see ``_QUEUE_DROP_BUDGET``.
_OUTBOUND_QUEUE_MAX = 256

# Consecutive drops on the same client before we proactively close it.
_QUEUE_DROP_BUDGET = 8

_SHUTDOWN_SENTINEL: bytes = b""

_T = TypeVar("_T")


@dataclass(frozen=True)
class ControlOptions:
    """Knobs for the remote control endpoint.

    ``port=0`` asks the OS for an ephemeral free port — useful for tests.
    ``allow_external`` flips the bind from loopback to all interfaces; combined
    with a missing ``token`` this is refused at startup.

    ``allow_port_fallback`` lets :meth:`NdjsonRpcEndpoint.start` retry once on an
    OS-assigned ephemeral port (0) when the requested port is already taken by an
    unrelated process. It is set only when the user did NOT pin a specific port:
    a default agreed-upon port that is busy falls back (the real port is then
    advertised via session discovery), while an explicitly-requested ``--control-port
    N`` that is busy fast-fails — pinning a port means the user wants *that* port.

    ``app_slug`` is the stable per-app discovery key (e.g. ``measure`` /
    ``fluxdep``), shared by the GUI writer and the MCP reader. Empty means the
    session is not advertised (no discovery file written).
    """

    port: int
    token: str | None = None
    allow_external: bool = False
    allow_port_fallback: bool = False
    app_slug: str = ""

    def host(self) -> str:
        return "0.0.0.0" if self.allow_external else "127.0.0.1"


class MainThreadDispatcher(QObject):
    """QObject living on the Qt main thread.

    Server threads ``emit`` ``invoke`` to schedule a callable on the main thread
    via Qt's queued connection — the supported way to marshal arbitrary work
    onto the Qt event loop from a foreign thread. Apps compose this into their
    own ``_dispatch_on_main`` (it is the byte-identical marshal primitive; the
    dispatch *policy* — guard / lifecycle / off-main — stays per-app).
    """

    invoke = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.invoke.connect(self._run, type=Qt.ConnectionType.QueuedConnection)  # type: ignore[call-arg]

    def _run(self, fn: Callable[[], None]) -> None:
        fn()


class ClientLink:
    """One accepted client's pure-transport state.

    Owns the recv buffer, the outbound queue + sole-writer thread, the drop
    counter and closing flag, and ``authed`` (the handshake the endpoint owns).
    It knows nothing of what a line *means*: app-specific per-connection state
    (event subscriptions, editor sessions) is attached by the router via
    :attr:`app_ctx`.
    """

    __slots__ = (
        "authed",
        "buffer",
        "peer",
        "outbound",
        "writer_thread",
        "consecutive_drops",
        "closing",
        "app_ctx",
    )

    def __init__(self, peer: str, token_required: bool) -> None:
        self.peer = peer
        # If no token is configured, every client starts authenticated.
        self.authed = not token_required
        self.buffer = bytearray()
        self.outbound: queue.Queue[bytes] = queue.Queue(maxsize=_OUTBOUND_QUEUE_MAX)
        self.writer_thread: threading.Thread | None = None
        self.consecutive_drops: int = 0
        self.closing: bool = False
        # The router attaches its per-connection semantic state here in
        # ``on_client_open`` (e.g. an event-subscription set); the endpoint
        # never reads it.
        self.app_ctx: object = None


class EndpointRouter(Protocol):
    """The seam each app implements to route parsed requests + own connections.

    The endpoint calls these; the app supplies everything downstream of the
    handshake.
    """

    def route(self, link: ClientLink, request: object) -> None:
        """Handle one parsed, authenticated, non-handshake request.

        Called on the IO/server thread. ``request`` is a
        :class:`zcu_tools.gui.remote.wire.Request`. The app decides the method
        set, validation, and how/where the handler runs (any main-thread
        marshal is the app's, via :class:`MainThreadDispatcher`). The app uses
        :meth:`NdjsonRpcEndpoint.reply_ok` / ``reply_error`` to respond.
        """
        ...

    def on_client_open(self, link: ClientLink) -> None:
        """A client connected — attach app-specific per-connection state.

        Runs on the IO/server thread (during accept). Typically sets
        ``link.app_ctx``.
        """
        ...

    def on_client_close(self, link: ClientLink, *, on_main_thread: bool) -> None:
        """A client is dropping — release any app-specific per-connection state.

        Called on drop (IO/server thread, ``on_main_thread=False``) and during
        ``stop()`` (Qt main thread, ``on_main_thread=True``). The flag lets a
        router that must touch main-thread-owned state (e.g. reclaim editor
        sessions) choose between marshalling and a direct call — the endpoint
        owns the thread context and reports it; the router owns the cleanup.
        """
        ...


class NdjsonRpcEndpoint:
    """Pure-transport NDJSON RPC server: socket + framing + handshake + push.

    Construct with the bind options, the wire/code version constants, a thread
    name prefix, and the app's :class:`EndpointRouter`. Inert until
    :meth:`start`.
    """

    def __init__(
        self,
        opts: ControlOptions,
        *,
        wire_version: int,
        gui_version: int,
        server_name: str,
        router: EndpointRouter,
    ) -> None:
        if opts.allow_external and not opts.token:
            raise RuntimeError(
                "Remote control must specify a token when allow_external is set"
            )
        self._opts = opts
        self._wire_version = wire_version
        self._gui_version = gui_version
        self._server_name = server_name
        self._router = router
        self._stopping = threading.Event()
        self._server_sock: socket.socket | None = None
        self._wake_r: socket.socket | None = None
        self._wake_w: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._actual_port: int | None = None
        self._clients: dict[socket.socket, ClientLink] = {}
        self._clients_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> int:
        """Bind, listen, spawn the server thread. Returns the actual bound port.

        The app subscribes its EventBus *before* calling this (so no event is
        missed); the endpoint only owns the socket side.
        """
        if self._thread is not None:
            raise RuntimeError("NdjsonRpcEndpoint.start() called twice")
        host = self._opts.host()
        sock = self._bind(host)
        sock.listen(8)
        sock.setblocking(False)
        self._server_sock = sock
        port = int(sock.getsockname()[1])
        self._actual_port = port

        wake_r, wake_w = socket.socketpair()
        wake_r.setblocking(False)
        self._wake_r, self._wake_w = wake_r, wake_w

        self._thread = threading.Thread(
            target=self._serve_forever, name=self._server_name, daemon=True
        )
        self._thread.start()
        logger.info(
            "%s listening on %s:%d (token=%s)",
            self._server_name,
            host,
            port,
            "yes" if self._opts.token else "no",
        )
        return port

    def _bind(self, host: str) -> socket.socket:
        """Bind the requested port; fall back to an ephemeral port if allowed.

        A free socket is returned. When the requested port is taken and
        ``allow_port_fallback`` is set (the default agreed-upon port, not a
        user-pinned one), retry once on port 0 so the OS hands out a free port —
        the real port is then advertised through session discovery. A pinned port
        (fallback off) fast-fails so the user's explicit choice is respected.
        """

        def _make() -> socket.socket:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s

        sock = _make()
        try:
            sock.bind((host, self._opts.port))
            return sock
        except OSError as exc:
            sock.close()
            if not (self._opts.allow_port_fallback and self._opts.port != 0):
                raise RuntimeError(
                    f"NdjsonRpcEndpoint bind {host}:{self._opts.port} failed: {exc}"
                ) from exc
            logger.warning(
                "%s: port %d is taken (%s); falling back to an ephemeral port",
                self._server_name,
                self._opts.port,
                exc,
            )
        sock = _make()
        try:
            sock.bind((host, 0))
            return sock
        except OSError as exc:
            sock.close()
            raise RuntimeError(
                f"NdjsonRpcEndpoint ephemeral-port bind on {host} failed: {exc}"
            ) from exc

    def stop(self) -> None:
        """Wake the selector loop, close all sockets, join threads.

        Idempotent. Must be called from the Qt main thread (it calls the
        router's ``on_client_close`` synchronously, which may touch app state).
        """
        if self._stopping.is_set():
            return
        self._stopping.set()

        with self._clients_lock:
            client_snapshot = list(self._clients.items())
        for _sock, link in client_snapshot:
            # Release app state on the main thread (stop() runs here) before
            # tearing the connection down.
            self._router.on_client_close(link, on_main_thread=True)
            link.closing = True
            try:
                link.outbound.put_nowait(_SHUTDOWN_SENTINEL)
            except queue.Full:
                try:
                    link.outbound.get_nowait()
                except queue.Empty:
                    pass
                try:
                    link.outbound.put_nowait(_SHUTDOWN_SENTINEL)
                except queue.Full:
                    pass
        for _sock, link in client_snapshot:
            t = link.writer_thread
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
        logger.info("%s stopped", self._server_name)

    @property
    def port(self) -> int:
        port = self._actual_port
        if port is None:
            raise RuntimeError("NdjsonRpcEndpoint.start() was not called")
        return port

    def has_live_client(self) -> bool:
        """Return True if at least one control client is currently connected.

        Thread-safe: reads _clients under the lock so it is safe to call from
        any thread (the router calls this on the IO thread; the main-thread
        refresh_feedback_widget reads it via the adapter façade).
        """
        with self._clients_lock:
            return bool(self._clients)

    def client_state_transaction(
        self, link: ClientLink, operation: Callable[[], _T]
    ) -> _T:
        """Run one app-owned per-client state operation under the registry lock.

        The endpoint does not inspect ``link.app_ctx``.  Routers use this narrow
        transaction to linearize subscription mutation/listing with lazy push
        recipient selection.  ``operation`` must be non-blocking and must not
        call back into the endpoint.
        """
        with self._clients_lock:
            return operation()

    # ------------------------------------------------------------------
    # Outbound: reply (to one link) + broadcast (push fan-out)
    # ------------------------------------------------------------------

    def reply_ok(self, link: ClientLink, *, rid: str, result) -> None:
        resp = Response(id=rid, ok=True, result=result)
        try:
            line = encode_line(resp.to_wire())
        except Exception:
            logger.exception("failed to encode reply for %s", rid)
            return
        self._enqueue(link, line, is_push=False)

    def reply_error(
        self,
        link: ClientLink,
        *,
        rid: str,
        code: ErrorCode,
        message: str,
        reason: str = "",
        data: dict | None = None,
    ) -> None:
        env = ErrorEnvelope(code=code.value, message=message, reason=reason, data=data)
        resp = Response(id=rid, ok=False, error=env)
        try:
            line = encode_line(resp.to_wire())
        except Exception:
            logger.exception("failed to encode error reply for %s", rid)
            return
        self._enqueue(link, line, is_push=False)

    def broadcast(self, line: bytes, predicate: Callable[[ClientLink], bool]) -> None:
        """Fan a pre-encoded push line out to every link passing ``predicate``.

        Pure mechanism: the endpoint does not know what the line is or why the
        predicate selects who it does. Called on the Qt main thread (every push
        originates from a main-thread event-bus callback / diagnostic fan-out).
        Reads ``_clients`` under the lock; ``_enqueue`` is thread-safe.
        """
        self.broadcast_lazy(lambda: line, predicate)

    def broadcast_lazy(
        self,
        line_factory: Callable[[], bytes | None],
        predicate: Callable[[ClientLink], bool],
        *,
        on_delivered: Callable[[ClientLink], None] | None = None,
    ) -> None:
        """Build one push line only when at least one live recipient needs it.

        Recipient selection is a two-phase transaction.  The first phase takes
        a registry-order snapshot of matching, non-closing links.  The caller's
        factory then runs outside ``_clients_lock`` (and therefore stays on the
        caller's thread).  The final phase revalidates the original recipients
        and enqueues under the same lock used by subscription mutation.  A
        subscribe after the first phase does not receive the old event; an
        unsubscribe or disconnect completed before final delivery receives no
        late push.

        ``predicate`` and ``on_delivered`` are app-owned O(1) state operations.
        They must not block or call back into the endpoint.  ``on_delivered`` is
        called only after the line was actually accepted by that link's queue.
        """
        with self._clients_lock:
            candidates = [
                (sock, link)
                for sock, link in self._clients.items()
                if not link.closing and predicate(link)
            ]
        if not candidates:
            return

        line = line_factory()
        if line is None:
            return

        with self._clients_lock:
            for sock, link in candidates:
                if (
                    self._clients.get(sock) is not link
                    or link.closing
                    or not predicate(link)
                ):
                    continue
                if not self._enqueue(link, line, is_push=True):
                    continue
                if on_delivered is not None:
                    on_delivered(link)

    def _enqueue(self, link: ClientLink, line: bytes, *, is_push: bool) -> bool:
        if link.closing:
            return False
        try:
            link.outbound.put_nowait(line)
            link.consecutive_drops = 0
            return True
        except queue.Full:
            link.consecutive_drops += 1
            logger.warning(
                "remote client %s outbound queue full (drops=%d, push=%s)",
                link.peer,
                link.consecutive_drops,
                is_push,
            )
            if link.consecutive_drops >= _QUEUE_DROP_BUDGET:
                logger.warning(
                    "remote client %s exceeded drop budget; closing", link.peer
                )
                link.closing = True
                try:
                    link.outbound.get_nowait()
                except queue.Empty:
                    pass
                try:
                    link.outbound.put_nowait(_SHUTDOWN_SENTINEL)
                except queue.Full:
                    pass
            return False

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
                            link = self._clients.get(sock)
                        if link is None:
                            sel.unregister(sock)
                            sock.close()
                            continue
                        self._service_client(sel, sock, link)
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
        link = ClientLink(peer=peer, token_required=token_required)
        self._router.on_client_open(link)
        with self._clients_lock:
            self._clients[csock] = link
        sel.register(csock, selectors.EVENT_READ, data=("client", peer))
        writer = threading.Thread(
            target=self._client_writer,
            args=(csock, link),
            name=f"{self._server_name}Writer[{peer}]",
            daemon=True,
        )
        link.writer_thread = writer
        writer.start()
        logger.info("remote client connected: %s", peer)

    def _client_writer(self, sock: socket.socket, link: ClientLink) -> None:
        """Drain the outbound queue to the socket; exits on sentinel / close."""
        while True:
            try:
                line = link.outbound.get(timeout=1.0)
            except queue.Empty:
                if link.closing or self._stopping.is_set():
                    return
                continue
            if line is _SHUTDOWN_SENTINEL or line == _SHUTDOWN_SENTINEL:
                return
            try:
                sock.sendall(line)
            except OSError as exc:
                logger.info("remote client writer %s exit on send: %s", link.peer, exc)
                link.closing = True
                return

    def _service_client(
        self, sel: selectors.BaseSelector, sock: socket.socket, link: ClientLink
    ) -> None:
        try:
            chunk = sock.recv(4096)
        except (BlockingIOError, InterruptedError):
            return
        except OSError as exc:
            logger.info("remote client recv error %s: %s", link.peer, exc)
            self._drop_client(sel, sock, link)
            return
        if not chunk:
            self._drop_client(sel, sock, link)
            return
        link.buffer.extend(chunk)
        while True:
            nl = link.buffer.find(LINE_TERMINATOR)
            if nl < 0:
                if len(link.buffer) > MAX_LINE_BYTES:
                    self.reply_error(
                        link,
                        rid="",
                        code=ErrorCode.INVALID_PARAMS,
                        message=f"request line exceeded {MAX_LINE_BYTES} bytes",
                    )
                    self._drop_client(sel, sock, link)
                return
            line = bytes(link.buffer[:nl])
            del link.buffer[: nl + 1]
            self._handle_line(link, line)

    def _drop_client(
        self, sel: selectors.BaseSelector, sock: socket.socket, link: ClientLink
    ) -> None:
        try:
            sel.unregister(sock)
        except (KeyError, ValueError):
            pass
        link.closing = True
        try:
            link.outbound.put_nowait(_SHUTDOWN_SENTINEL)
        except queue.Full:
            pass
        try:
            sock.close()
        except OSError:
            pass
        with self._clients_lock:
            self._clients.pop(sock, None)
        # Let the app release per-connection state (editor sessions etc.). On a
        # drop this runs on the IO thread; the router marshals to the main
        # thread itself if its cleanup must.
        self._router.on_client_close(link, on_main_thread=False)
        logger.info("remote client disconnected: %s", link.peer)

    # ------------------------------------------------------------------
    # Per-request handling: built-in handshake, then hand off to the router
    # ------------------------------------------------------------------

    def _handle_line(self, link: ClientLink, line: bytes) -> None:
        rid = ""
        try:
            raw = decode_line(line)
            req = parse_request(raw)
            rid = req.id
            # wire.version is a no-auth handshake probe: it lets a caller read
            # the server's wire-contract version + GUI code revision before
            # authenticating, so an incompatible contract or a stale GUI process
            # is detectable on connect.
            if req.method == "wire.version":
                self.reply_ok(
                    link,
                    rid=req.id,
                    result={
                        "wire_version": self._wire_version,
                        "gui_version": self._gui_version,
                    },
                )
                return
            if req.method == "auth":
                self._handle_auth(link, req.id, req.params)
                return
            if not link.authed:
                self.reply_error(
                    link,
                    rid=req.id,
                    code=ErrorCode.UNAUTHORIZED,
                    message="auth required: send {method: 'auth', params: {token: ...}} first",
                )
                return
            # Past the handshake: everything else is the app's. The router owns
            # the method set, validation, and dispatch (incl. any main-thread
            # marshal); the endpoint is zero-knowledge of method semantics.
            self._router.route(link, req)
        except RemoteError as exc:
            self.reply_error(
                link,
                rid=rid,
                code=exc.code,
                message=exc.message,
                reason=exc.reason,
                data=exc.data,
            )

    def _handle_auth(self, link: ClientLink, rid: str, params) -> None:
        token = require_str(params, "token")
        configured = self._opts.token
        if not configured:
            self.reply_error(
                link,
                rid=rid,
                code=ErrorCode.PRECONDITION_FAILED,
                message="auth disabled on this server",
            )
            return
        if hmac.compare_digest(token, configured):
            link.authed = True
            self.reply_ok(link, rid=rid, result={})
        else:
            self.reply_error(
                link, rid=rid, code=ErrorCode.UNAUTHORIZED, message="invalid token"
            )


__all__ = [
    "ClientLink",
    "ControlOptions",
    "EndpointRouter",
    "MainThreadDispatcher",
    "NdjsonRpcEndpoint",
]
