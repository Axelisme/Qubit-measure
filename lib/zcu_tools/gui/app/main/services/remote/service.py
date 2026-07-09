"""RemoteControlAdapter — measure-gui's second View (driving adapter).

The RPC face onto the Controller, peer to the Qt ``MainWindow`` (ADR-0005 /
ADR-0013): the second user-facing client (user = an automation agent / another
server). The shared dispatch scaffolding (the EndpointRouter seam, the
main-thread marshal, the EventBus push fan-out) lives in
:class:`RemoteControlServiceBase`; this is the *richest* of the three apps and
layers measure-gui's own dispatch policy on top by overriding the base seams:

  - ``_new_client_ctx`` → a :class:`_ClientCtx` that also tracks CfgEditor
    sessions; ``_on_client_close_extra`` reclaims them on drop;
  - ``_route_extra`` → the editor.subscribe/unsubscribe state-owning methods;
  - ``_guard`` → the optimistic version guard (run on the main thread inside the
    handler ``_run`` so its compare-and-act is atomic); ``_after_success`` →
    CfgEditor session lifecycle tracking;
  - ``_extra_start`` / ``_extra_stop`` → the per-editor change stream and the
    out-of-band diagnostic channel (``notify_diagnostic``), both pushed via
    ``endpoint.broadcast`` independent of EventBus.

Handlers receive *this adapter* (not the bare ctrl), so they reach tab-resource
commands through ``adapter.tab_control``, other app commands through
``adapter.ctrl.<façade>``, context commands through
``adapter.context_control``, device commands through ``adapter.device_control``,
predictor commands through ``adapter.predictor_control``, and View-side surfaces
(render/snapshot) through ``adapter.render_view``. Construct after ``Controller``
/ ``MainWindow`` exist; inert until ``start()``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

from zcu_tools.gui.remote.control_service import (
    ControlOptions,
    RemoteControlServiceBase,
    SubscriptionCtx,
)
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.remote.framing import encode_line
from zcu_tools.gui.remote.rpc_endpoint import ClientLink
from zcu_tools.gui.remote.wire import Request

if TYPE_CHECKING:
    # Type-only: importing Controller at runtime would form a cycle
    # (controller.py imports remote.dialogs). String annotation keeps pyright
    # checking handler/ctrl method names while the runtime import never happens.
    from zcu_tools.gui.app.main.controller import Controller, RenderView, Severity
    from zcu_tools.gui.app.main.services.tab_control import TabControlPort
    from zcu_tools.gui.session.context_control import ContextControlPort
    from zcu_tools.gui.session.device_control import DeviceControlPort
    from zcu_tools.gui.session.predictor_control import PredictorControlPort

from .dispatch import METHOD_REGISTRY
from .events import EVENT_SERIALIZERS, wire_event_name
from .wire_version import GUI_VERSION, WIRE_VERSION

logger = logging.getLogger(__name__)


class _ClientCtx(SubscriptionCtx):
    """measure-gui's per-connection semantic state (attached to ``link.app_ctx``).

    Extends the base subscription set with the CfgEditor sessions this connection
    owns (reclaimed on drop) and subscribed to (for the per-editor change stream).
    """

    __slots__ = ("editor_ids", "subscribed_editors")

    def __init__(self) -> None:
        super().__init__()
        # CfgEditor session ids opened by this connection; reclaimed on drop.
        self.editor_ids: set[str] = set()
        # CfgEditor session ids this connection subscribed to for change push.
        self.subscribed_editors: set[str] = set()


def _ctx(link: ClientLink) -> _ClientCtx:
    ctx = link.app_ctx
    assert isinstance(ctx, _ClientCtx)
    return ctx


class RemoteControlAdapter(RemoteControlServiceBase):
    """Driving adapter: an NDJSON RPC face onto the measure-gui ``Controller``.

    Holds the concrete ``Controller`` (app command face), exposes the shared
    context/device/predictor-control facets, exposes the app-local tab-control
    facet, and pulls EventBus from it via
    ``get_bus()``. Dispatch handlers reach tab commands through ``adapter.tab_control``,
    other app commands through ``adapter.ctrl``,
    context commands through ``adapter.context_control``, device commands through
    ``adapter.device_control``, predictor commands through ``adapter.predictor_control``,
    and the canvas-bearing View's pure-read surface through ``adapter.render_view``
    (screenshot / snapshot / dialog).
    ``render_view`` is None in a headless process; render handlers fail-fast then.
    """

    ctrl: Controller
    tab_control: TabControlPort
    context_control: ContextControlPort
    device_control: DeviceControlPort
    predictor_control: PredictorControlPort

    def __init__(
        self,
        controller: Controller,
        opts: ControlOptions,
        render_view: RenderView | None = None,
    ) -> None:
        super().__init__(
            controller,
            opts,
            wire_version=WIRE_VERSION,
            gui_version=GUI_VERSION,
            server_name="RemoteControlServer",
            method_registry=METHOD_REGISTRY,
            event_serializers=EVENT_SERIALIZERS,
            wire_event_name=wire_event_name,
        )
        self.render_view = render_view
        self.tab_control = controller.tab_control
        self.context_control = controller.context_control
        self.device_control = controller.device_control
        self.predictor_control = controller.predictor_control

    # ------------------------------------------------------------------
    # Base seams
    # ------------------------------------------------------------------

    def _new_client_ctx(self) -> SubscriptionCtx:
        return _ClientCtx()

    def _get_bus(self):
        return self.ctrl.get_bus()

    def _extra_start(self) -> None:
        self._wire_editor_change_listener()
        # Become a diagnostic-only View (ADR-0013): receive ctrl error/info
        # fan-out and push it to clients out-of-band of EventBus.
        self.ctrl.add_diagnostic_sink(self)
        # Inject has_live_client so the Controller (and via it, MainWindow) can
        # gate the feedback widget on agent presence (ADR-0025 C3).
        self.ctrl.set_agent_connected_query(self.has_live_client)

    def _extra_stop(self) -> None:
        self._unwire_editor_change_listener()
        self.ctrl.remove_diagnostic_sink(self)
        # Remove the predicate so has_agent_connected() returns False after stop.
        self.ctrl.set_agent_connected_query(None)

    def _on_client_count_changed(self) -> None:
        # Called on the Qt main thread whenever a client connects or disconnects.
        # Re-evaluate widget visibility (ADR-0025 C3: show only when op live AND
        # agent connected). Delegated through the RenderView Protocol so the
        # adapter is not coupled to MainWindow's private layout methods.
        if self.render_view is not None:
            self.render_view.refresh_feedback_widget()

    def _route_extra(self, link: ClientLink, req: Request) -> bool:
        # editor.subscribe/unsubscribe are state-owning (per-connection editor
        # subscription set), so handled here, not via dispatch.
        if req.method == "editor.subscribe":
            self._handle_editor_subscribe(link, req.id, req.params, subscribe=True)
            return True
        if req.method == "editor.unsubscribe":
            self._handle_editor_subscribe(link, req.id, req.params, subscribe=False)
            return True
        return False

    def _on_client_close_extra(
        self, ctx: SubscriptionCtx, *, on_main_thread: bool
    ) -> None:
        # Reclaim this connection's CfgEditor sessions. On a drop (IO thread) the
        # LiveModel teardown must be marshalled onto the Qt main thread; during
        # stop() the endpoint already calls us there, so reclaim directly.
        assert isinstance(ctx, _ClientCtx)
        self._reclaim_editors(ctx, marshal=not on_main_thread)

    # ------------------------------------------------------------------
    # editor.* state-owning handler
    # ------------------------------------------------------------------

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
    # Dispatch policy seams: version guard + editor lifecycle
    # ------------------------------------------------------------------

    def _guard(self, params: Mapping[str, object]) -> None:
        # Base dispatch seam → measure-gui's named version-guard policy.
        self._guard_versions(params)

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

    def _after_success(
        self,
        ctx: SubscriptionCtx,
        method: str,
        params: Mapping[str, object],
        result: Mapping[str, object],
    ) -> None:
        # Base dispatch seam → measure-gui's named editor-lifecycle bookkeeping.
        assert isinstance(ctx, _ClientCtx)
        self._track_editor_lifecycle(ctx, method, params, result)

    def _track_editor_lifecycle(
        self,
        ctx: _ClientCtx,
        method: str,
        params: Mapping[str, object],
        result: object,
    ) -> None:
        """Record/forget CfgEditor session ids per connection.

        ``editor.new`` binds the returned id to this client so a disconnect
        reclaims it; ``commit``/``discard`` forget it (the session is already
        gone server-side). Runs on the IO thread, where ``ctx.editor_ids`` lives.
        """
        if method == "editor.new":
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
    # Diagnostic channel (DiagnosticSink impl) — independent of EventBus
    # ------------------------------------------------------------------

    def notify_diagnostic(self, severity: Severity, title: str, message: str) -> None:
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


__all__ = ["ControlOptions", "RemoteControlAdapter"]
