"""RemoteControlAdapter — measure-gui's second View (driving adapter).

The RPC face onto the Controller, peer to the Qt ``MainWindow`` (ADR-0008 /
ADR-0013): the second user-facing client (user = an automation agent / another
server). Pure transport — the socket lifecycle, NDJSON framing, the per-client
writer, the ``wire.version`` / ``auth`` handshakes, and the push fan-out
primitive — lives in the shared :class:`NdjsonRpcEndpoint`. This adapter is the
richest of the three; it keeps all of measure-gui's *dispatch policy + domain*:

  - the :class:`EndpointRouter` seam: ``route`` (events.* + editor.* state-owning
    handlers, then METHOD_REGISTRY lookup + ParamSpec validation + dispatch),
    ``on_client_open`` (per-connection subscription + editor-session sets),
    ``on_client_close`` (reclaim this connection's CfgEditor sessions);
  - ``_dispatch_on_main``: the marshal (via the shared
    :class:`MainThreadDispatcher`) composed with measure-gui's own policy — the
    ``off_main_thread`` blocking-handler branch, the version guard
    (``_guard_versions``, run inside the main-thread ``_run`` so its
    compare-and-act is atomic), and ``_track_editor_lifecycle``;
  - EventBus push (keyed by :class:`GuiEvent`), the out-of-band diagnostic
    channel (``notify_diagnostic``), and the per-editor change stream
    (``_on_editor_event``) — all pushed via ``endpoint.broadcast``.

Handlers receive *this adapter* (not the bare ctrl), so they reach commands
through ``adapter.ctrl.<façade>`` and View-side surfaces (render/snapshot)
through ``adapter.render_view``. Construct after ``Controller`` / ``MainWindow``
exist; inert until ``start()``.
"""

from __future__ import annotations

import base64
import logging
import threading
from typing import TYPE_CHECKING, Callable, Mapping, Optional

from zcu_tools.gui.app.main.event_bus import EventBus, GuiEvent, Payload
from zcu_tools.gui.remote.control_adapter import (
    ClientLink,
    ControlOptions,
    MainThreadDispatcher,
    NdjsonRpcEndpoint,
)
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.remote.framing import encode_line
from zcu_tools.gui.remote.param_spec import validate_params
from zcu_tools.gui.remote.wire import Request

if TYPE_CHECKING:
    # Type-only: importing Controller at runtime would form a cycle
    # (controller.py imports remote.dialogs). String annotation keeps pyright
    # checking handler/ctrl method names while the runtime import never happens.
    from zcu_tools.gui.app.main.controller import Controller, RenderView, Severity

from .dispatch import METHOD_REGISTRY
from .events import EVENT_SERIALIZERS, wire_event_name
from .wire_version import GUI_VERSION, WIRE_VERSION

logger = logging.getLogger(__name__)


class _ClientCtx:
    """measure-gui's per-connection semantic state (attached to ``link.app_ctx``).

    Tracks event subscriptions plus the CfgEditor sessions this connection owns
    (reclaimed on drop) and subscribed to (for the per-editor change stream).
    """

    __slots__ = ("subscribed", "editor_ids", "subscribed_editors")

    def __init__(self) -> None:
        self.subscribed: set[str] = set()
        # CfgEditor session ids opened by this connection; reclaimed on drop.
        self.editor_ids: set[str] = set()
        # CfgEditor session ids this connection subscribed to for change push.
        self.subscribed_editors: set[str] = set()


def _ctx(link: ClientLink) -> _ClientCtx:
    ctx = link.app_ctx
    assert isinstance(ctx, _ClientCtx)
    return ctx


class RemoteControlAdapter:
    """Driving adapter: an NDJSON RPC face onto the measure-gui ``Controller``.

    Holds the concrete ``Controller`` (command face) and a
    :class:`NdjsonRpcEndpoint` (transport); pulls EventBus from the controller
    via ``get_bus()``. Dispatch handlers reach commands through ``adapter.ctrl``
    and the canvas-bearing View's pure-read surface through ``adapter.render_view``
    (screenshot / snapshot / dialog). ``render_view`` is None in a headless
    process; render handlers fail-fast then.
    """

    def __init__(
        self,
        controller: "Controller",
        opts: ControlOptions,
        render_view: Optional["RenderView"] = None,
    ) -> None:
        self.ctrl = controller
        self.render_view = render_view
        self._dispatcher = MainThreadDispatcher()
        self._endpoint = NdjsonRpcEndpoint(
            opts,
            wire_version=WIRE_VERSION,
            gui_version=GUI_VERSION,
            server_name="RemoteControlServer",
            router=self,
        )
        # EventBus subscriptions registered in start(); unsubscribed in stop().
        self._bus: Optional[EventBus] = None
        self._bus_subs: list[tuple[GuiEvent, Callable[[Payload], None]]] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> int:
        """Hook EventBus + editor stream + diagnostic sink, then start the endpoint.

        Returns the bound port. The app-side wiring happens before the socket
        opens so no event/diagnostic is missed.
        """
        self._subscribe_event_bus()
        self._wire_editor_change_listener()
        # Become a diagnostic-only View (ADR-0013): receive ctrl error/info
        # fan-out and push it to clients out-of-band of EventBus.
        self.ctrl.add_diagnostic_sink(self)
        return self._endpoint.start()

    def stop(self) -> None:
        """Unwire app-side listeners, then stop the endpoint. Idempotent. Main thread.

        Unsubscribe EventBus + editor change stream + diagnostic sink first so no
        more enqueues happen; the endpoint then reclaims editors (via
        ``on_client_close``), drains writers, and closes sockets.
        """
        self._unsubscribe_event_bus()
        self._unwire_editor_change_listener()
        self.ctrl.remove_diagnostic_sink(self)
        self._endpoint.stop()

    @property
    def port(self) -> int:
        return self._endpoint.port

    # ------------------------------------------------------------------
    # EndpointRouter seam
    # ------------------------------------------------------------------

    def on_client_open(self, link: ClientLink) -> None:
        link.app_ctx = _ClientCtx()

    def on_client_close(self, link: ClientLink, *, on_main_thread: bool) -> None:
        # Reclaim this connection's CfgEditor sessions. On a drop (IO thread) the
        # LiveModel teardown must be marshalled onto the Qt main thread; during
        # stop() the endpoint already calls us there, so reclaim directly.
        self._reclaim_editors(_ctx(link), marshal=not on_main_thread)

    def route(self, link: ClientLink, request: object) -> None:
        """Handle one parsed, authenticated request on the IO thread."""
        assert isinstance(request, Request)
        req = request
        # Subscription methods are state-owning (per-connection sets), so handled
        # here, not via dispatch.
        if req.method == "events.subscribe":
            self._handle_subscribe(link, req.id, req.params)
            return
        if req.method == "events.unsubscribe":
            self._handle_unsubscribe(link, req.id, req.params)
            return
        if req.method == "events.list":
            self._endpoint.reply_ok(
                link,
                rid=req.id,
                result={
                    "events": sorted(wire_event_name(e) for e in EVENT_SERIALIZERS),
                    "subscribed": sorted(_ctx(link).subscribed),
                },
            )
            return
        # editor.subscribe/unsubscribe are state-owning (per-connection editor
        # subscription set), so handled here, not via dispatch.
        if req.method == "editor.subscribe":
            self._handle_editor_subscribe(link, req.id, req.params, subscribe=True)
            return
        if req.method == "editor.unsubscribe":
            self._handle_editor_subscribe(link, req.id, req.params, subscribe=False)
            return
        spec = METHOD_REGISTRY.get(req.method)
        if spec is None:
            self._endpoint.reply_error(
                link,
                rid=req.id,
                code=ErrorCode.UNKNOWN_METHOD,
                message=f"unknown method: {req.method!r}",
            )
            return
        # Validate params against the method's ParamSpec contract on the IO
        # thread (pure, no Qt needed) so malformed requests fail fast without
        # consuming a main-thread hop. Methods that have not declared params yet
        # pass their raw params through unchanged.
        if spec.params:
            handler_params = validate_params(spec.params, req.params)
        else:
            handler_params = req.params
        self._dispatch_on_main(link, req.id, req.method, spec, handler_params)

    # ------------------------------------------------------------------
    # events.* / editor.* state-owning handlers
    # ------------------------------------------------------------------

    def _handle_subscribe(self, link: ClientLink, rid: str, params) -> None:
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
        ctx = _ctx(link)
        for ev in events:
            assert isinstance(ev, str)
            ctx.subscribed.add(ev)
        self._endpoint.reply_ok(
            link, rid=rid, result={"subscribed": sorted(ctx.subscribed)}
        )

    def _handle_unsubscribe(self, link: ClientLink, rid: str, params) -> None:
        events = params.get("events")
        if not isinstance(events, list):
            raise RemoteError(
                ErrorCode.INVALID_PARAMS, "'events' must be a list of event names"
            )
        ctx = _ctx(link)
        for ev in events:
            if isinstance(ev, str):
                ctx.subscribed.discard(ev)
        self._endpoint.reply_ok(
            link, rid=rid, result={"subscribed": sorted(ctx.subscribed)}
        )

    def _handle_editor_subscribe(
        self, link: ClientLink, rid: str, params, *, subscribe: bool
    ) -> None:
        editor_id = params.get("editor_id")
        if not isinstance(editor_id, str) or not editor_id:
            raise RemoteError(
                ErrorCode.INVALID_PARAMS, "'editor_id' must be a non-empty string"
            )
        # No existence check: subscription is a pure per-connection filter. A
        # client may subscribe before/around open; pushes only flow for live
        # sessions, and editor_closed cleans the set.
        ctx = _ctx(link)
        if subscribe:
            ctx.subscribed_editors.add(editor_id)
        else:
            ctx.subscribed_editors.discard(editor_id)
        self._endpoint.reply_ok(
            link,
            rid=rid,
            result={"subscribed_editors": sorted(ctx.subscribed_editors)},
        )

    # ------------------------------------------------------------------
    # Dispatch: marshal + measure-gui policy (off-main / guard / lifecycle)
    # ------------------------------------------------------------------

    def _dispatch_on_main(self, link: ClientLink, rid, method, spec, params) -> None:
        holder: dict[str, object] = {}

        if spec.off_main_thread:
            # Blocking handler (operation.await): run on THIS IO worker thread,
            # never the main thread — marshalling it onto the main thread would
            # deadlock (it would occupy the event loop that must dispatch the
            # worker signal it awaits). It must only do thread-safe waiting and
            # must not touch the version guard / editor lifecycle, so none of
            # those are run here.
            try:
                holder["result"] = spec.handler(self, params)
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
                self._endpoint.reply_error(
                    link,
                    rid=rid,
                    code=ErrorCode.TIMEOUT,
                    message=f"handler did not complete within {spec.timeout_seconds}s",
                )
                return
        if "remote_error" in holder:
            exc = holder["remote_error"]
            assert isinstance(exc, RemoteError)
            self._endpoint.reply_error(
                link,
                rid=rid,
                code=exc.code,
                message=exc.message,
                reason=exc.reason,
                data=exc.data,
            )
            return
        if "controller_error" in holder:
            err = holder["controller_error"]
            self._endpoint.reply_error(
                link, rid=rid, code=ErrorCode.CONTROLLER_ERROR, message=str(err)
            )
            return
        # Every handler returns a wire dict (Mapping[str, object]); guard the
        # handler-return invariant (this is the result, not a ParamSpec-validated
        # input, so the check is not redundant with wire validation).
        result = holder["result"]
        assert isinstance(result, dict), f"handler {method!r} returned non-dict result"
        self._track_editor_lifecycle(_ctx(link), method, params, result)
        self._endpoint.reply_ok(link, rid=rid, result=result)

    def _guard_versions(self, params: Mapping[str, object]) -> None:
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
        expected = params.get("expected_versions")
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
            # agent-facing message). It DOES carry the resource *identities* that
            # moved (data.stale), so mcp can name them in agent language.
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
                data={"stale": sorted(mismatched.keys())},
            )

    def _ctrl_resource_versions(self) -> dict[str, int]:
        return dict(self.ctrl.resources_versions())

    def _safe_editor_id_for_owner(self, owner_key: str) -> Optional[str]:
        return self.ctrl.editor_id_for_owner(owner_key)

    def _track_editor_lifecycle(
        self,
        ctx: _ClientCtx,
        method: str,
        params: Mapping[str, object],
        result: object,
    ) -> None:
        """Record/forget CfgEditor session ids per connection.

        ``editor.open`` binds the returned id to this client so a disconnect
        reclaims it; ``commit``/``discard`` forget it (the session is already
        gone server-side). Runs on the IO thread, where ``ctx.editor_ids`` lives.
        """
        if method == "editor.open":
            editor_id = result.get("editor_id") if isinstance(result, dict) else None
            if isinstance(editor_id, str):
                ctx.editor_ids.add(editor_id)
        elif method in ("editor.commit", "editor.discard"):
            editor_id = params.get("editor_id")
            if isinstance(editor_id, str):
                ctx.editor_ids.discard(editor_id)

    def _reclaim_editors(self, ctx: _ClientCtx, *, marshal: bool) -> None:
        """Discard CfgEditor sessions opened by ``ctx``; clears its id set.

        ``marshal=True`` schedules the discard on the Qt main thread (use when
        called from the IO/server thread, i.e. a client drop); ``marshal=False``
        calls directly (use from the main thread, i.e. ``stop``).
        """
        ids = list(ctx.editor_ids)
        ctx.editor_ids.clear()
        if not ids:
            return

        def _run() -> None:
            try:
                self.ctrl.discard_cfg_editors(ids)
            except Exception:  # pragma: no cover — best-effort cleanup
                logger.exception("failed to reclaim editor sessions %r", ids)

        if marshal:
            self._dispatcher.invoke.emit(_run)
        else:
            _run()

    # ------------------------------------------------------------------
    # EventBus integration (subscribe on main thread; push via broadcast)
    # ------------------------------------------------------------------

    def _subscribe_event_bus(self) -> None:
        """Subscribe one callback per serialised GuiEvent on the main thread."""
        bus = self.ctrl.get_bus()
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
            "event-flow: subscribed %d EventBus events for push: %s",
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
            self._endpoint.broadcast(
                line, predicate=lambda link: wire_name in _ctx(link).subscribed
            )

        return _on_event

    # ------------------------------------------------------------------
    # Diagnostic channel (DiagnosticSink impl) — independent of EventBus
    # ------------------------------------------------------------------

    def notify_diagnostic(self, severity: "Severity", title: str, message: str) -> None:
        """Push a Controller diagnostic to every client. Runs on the Qt main
        thread (ctrl fans out there). Deliberately *not* gated by event
        subscription and *not* routed through EventBus — diagnostics must reach
        the agent regardless of what it subscribed to, and a channel that
        reports a fault must not be the faulty channel (ADR-0013)."""
        try:
            line = encode_line(
                {
                    "event": "diagnostic",
                    "payload": {
                        "severity": severity,
                        "title": title,
                        "message": message,
                    },
                }
            )
        except Exception:  # pragma: no cover — payload is plain strings
            logger.exception("failed to encode diagnostic %r/%r", severity, title)
            return
        # Diagnostics reach every client, regardless of subscription.
        self._endpoint.broadcast(line, predicate=lambda link: True)

    # ------------------------------------------------------------------
    # CfgEditor per-session change stream (independent of EventBus)
    # ------------------------------------------------------------------

    def _wire_editor_change_listener(self) -> None:
        """Inject ``_on_editor_event`` into the CfgEditorService (via ctrl)."""
        self.ctrl.set_cfg_editor_change_listener(self._on_editor_event)

    def _unwire_editor_change_listener(self) -> None:
        self.ctrl.set_cfg_editor_change_listener(None)

    def _on_editor_event(self, editor_id: str, event_name: str, payload: dict) -> None:
        """Push a per-editor notification. Runs on the Qt main thread.

        ``event_name`` ∈ {editor_changed, editor_closed}. Only clients that
        subscribed to ``editor_id`` receive it. On editor_closed we also drop the
        id from every client's subscription set. (The editor's resource version
        is bumped at the edit site, not here.)
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
        closing = event_name == "editor_closed"

        def _predicate(link: ClientLink) -> bool:
            ctx = _ctx(link)
            if editor_id not in ctx.subscribed_editors:
                return False
            # On editor_closed, drop the id from this client's set as we push.
            if closing:
                ctx.subscribed_editors.discard(editor_id)
            return True

        self._endpoint.broadcast(line, predicate=_predicate)


# Re-export base64 helpers used by view.screenshot.
__all__ = ["ControlOptions", "RemoteControlAdapter", "base64"]
