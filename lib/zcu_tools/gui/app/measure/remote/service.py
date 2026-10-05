"""RemoteControlAdapter — measure-gui's second View (driving adapter).

The RPC face onto the Controller, peer to the Qt ``MainWindow`` (ADR-0067 /
ADR-0068): the second user-facing client (user = an automation agent / another
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
commands through ``adapter.tab_control``, run/analyze commands through
``adapter.run_analyze_control``, generic operation await/progress through
``adapter.operation_control``, save commands through ``adapter.save_control``,
writeback commands through ``adapter.writeback_control``,
other app commands through
``adapter.ctrl.<façade>``, context commands through
``adapter.context_control``, device commands through ``adapter.device_control``,
predictor commands through ``adapter.predictor_control``, and View-side surfaces
(render/snapshot) through ``adapter.render_view``. Construct after ``Controller``
/ ``MainWindow`` exist; inert until ``start()``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, cast

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
    from zcu_tools.gui.app.measure.controller import Controller, RenderView, Severity
    from zcu_tools.gui.app.measure.services.operation_control import (
        OperationControlPort,
    )
    from zcu_tools.gui.app.measure.services.run_analyze_control import (
        RunAnalyzeControlPort,
    )
    from zcu_tools.gui.app.measure.services.save_control import SaveControlPort
    from zcu_tools.gui.app.measure.services.tab_control import TabControlPort
    from zcu_tools.gui.app.measure.services.writeback_control import (
        WritebackControlPort,
    )
    from zcu_tools.gui.cfg.binding import SettableTarget
    from zcu_tools.gui.cfg.resource import CfgEditing
    from zcu_tools.gui.session.context_control import ContextControlPort
    from zcu_tools.gui.session.device_control import DeviceControlPort
    from zcu_tools.gui.session.ports import OwnerScheduler
    from zcu_tools.gui.session.predictor_control import PredictorControlPort

from .dispatch import METHOD_REGISTRY
from .events import EVENT_SERIALIZERS, wire_event_name
from .method_entries import METHOD_ENTRIES
from .method_entries._registry import AgentMethodPolicy
from .wire_version import GUI_VERSION, WIRE_VERSION

logger = logging.getLogger(__name__)


class _ClientCtx(SubscriptionCtx):
    """measure-gui's per-connection semantic state (attached to ``link.app_ctx``).

    Extends the base subscription set with the CfgEditor sessions this connection
    owns (reclaimed on drop) and subscribed to (for the per-editor change stream).
    """

    __slots__ = ("editor_ids", "subscribed_editors", "seen")

    def __init__(self) -> None:
        super().__init__()
        # CfgEditor session ids opened by this connection; reclaimed on drop.
        self.editor_ids: set[str] = set()
        # CfgEditor session ids this connection subscribed to for change push.
        self.subscribed_editors: set[str] = set()
        # Only accessed by the owner-thread guard and observation hooks.
        self.seen: dict[str, int] = {}


def _ctx(link: ClientLink) -> _ClientCtx:
    ctx = link.app_ctx
    assert isinstance(ctx, _ClientCtx)
    return ctx


class RemoteControlAdapter(RemoteControlServiceBase):
    """Driving adapter: an NDJSON RPC face onto the measure-gui ``Controller``.

    Holds the concrete ``Controller`` (app command face), exposes the shared
    context/device/predictor-control facets, exposes the app-local tab-control
    run/analyze-control, operation-control, save-control, and writeback-control facets, and pulls EventBus from it via
    ``get_bus()``. Dispatch handlers reach tab commands through ``adapter.tab_control``,
    run/analyze commands through ``adapter.run_analyze_control``, operation
    await/progress through ``adapter.operation_control``, save commands through
    ``adapter.save_control``, writeback commands through
    ``adapter.writeback_control``, other app commands through ``adapter.ctrl``,
    context commands through ``adapter.context_control``, device commands through
    ``adapter.device_control``, predictor commands through ``adapter.predictor_control``,
    and the canvas-bearing View's pure-read surface through ``adapter.render_view``
    (screenshot / snapshot / dialog).
    ``render_view`` is None in a headless process; render handlers fail-fast then.
    """

    ctrl: Controller
    tab_control: TabControlPort
    cfg_lookup: Callable[[str], CfgEditing]
    run_analyze_control: RunAnalyzeControlPort
    operation_control: OperationControlPort
    save_control: SaveControlPort
    writeback_control: WritebackControlPort
    context_control: ContextControlPort
    device_control: DeviceControlPort
    predictor_control: PredictorControlPort

    def __init__(
        self,
        controller: Controller,
        opts: ControlOptions,
        owner_scheduler: OwnerScheduler,
        render_view: RenderView | None = None,
    ) -> None:
        super().__init__(
            controller,
            opts,
            owner_scheduler=owner_scheduler,
            wire_version=WIRE_VERSION,
            gui_version=GUI_VERSION,
            server_name="RemoteControlServer",
            method_registry=METHOD_REGISTRY,
            event_serializers=EVENT_SERIALIZERS,
            wire_event_name=wire_event_name,
        )
        self.render_view = render_view
        self.tab_control = controller.tab_control
        self.cfg_lookup = controller.cfg_resources.lookup
        self.run_analyze_control = controller.run_analyze_control
        self.operation_control = controller.operation_control
        self.save_control = controller.save_control
        self.writeback_control = controller.writeback_control
        self.context_control = controller.context_control
        self.device_control = controller.device_control
        self.predictor_control = controller.predictor_control
        self._agent_policies: dict[str, AgentMethodPolicy] = {
            entry.method: entry.agent for entry in METHOD_ENTRIES
        }

    # ------------------------------------------------------------------
    # Base seams
    # ------------------------------------------------------------------

    def _new_client_ctx(self) -> SubscriptionCtx:
        return _ClientCtx()

    def _get_bus(self):
        return self.ctrl.get_bus()

    def _extra_start(self) -> None:
        self._wire_editor_change_listener()
        # Become a diagnostic-only View (ADR-0068): receive ctrl error/info
        # fan-out and push it to clients out-of-band of EventBus.
        self.ctrl.add_diagnostic_sink(self)

    def _extra_stop(self) -> None:
        self._unwire_editor_change_listener()
        self.ctrl.remove_diagnostic_sink(self)

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
        self, ctx: SubscriptionCtx, *, on_owner_thread: bool
    ) -> None:
        # No further owner request may use this ctx after the link closes.
        # The per-connection seen map is released with the ctx, not copied elsewhere.
        # Reclaim this connection's CfgEditor sessions. On a drop (IO thread) the
        # LiveModel teardown must be marshalled onto the State owner thread; during
        # stop() the endpoint already calls us there, so reclaim directly.
        assert isinstance(ctx, _ClientCtx)
        self._reclaim_editors(ctx, marshal=not on_owner_thread)

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
        def _update() -> list[str]:
            ctx = _ctx(link)
            if subscribe:
                ctx.subscribed_editors.add(editor_id)
            else:
                ctx.subscribed_editors.discard(editor_id)
            return sorted(ctx.subscribed_editors)

        subscribed_editors = self._endpoint.client_state_transaction(link, _update)
        self._endpoint.reply_ok(
            link,
            rid=rid,
            result={"subscribed_editors": subscribed_editors},
        )

    # ------------------------------------------------------------------
    # Dispatch policy seams: version guard + editor lifecycle
    # ------------------------------------------------------------------

    @staticmethod
    def _resource_key(template: str, values: Mapping[str, object]) -> str:
        if "{writeback_resource}" in template:
            pane = values.get("subtab_id")
            if pane not in ("analysis", "post_analysis"):
                raise RemoteError(ErrorCode.INVALID_PARAMS, "invalid writeback pane")
            values = dict(
                values,
                writeback_resource=(
                    "analyze" if pane == "analysis" else "post_analyze"
                ),
            )
        return template.format_map(values)

    def _guard(
        self, ctx: SubscriptionCtx, method: str, params: Mapping[str, object]
    ) -> None:
        assert isinstance(ctx, _ClientCtx)
        deps = self._agent_policies[method].guard_deps
        if not deps:
            return
        current = self._ctrl_resource_versions()
        required: set[str] = set()
        for template in deps:
            if template == "device:*":
                required.update(key for key in current if key.startswith("device:"))
                required.update(key for key in ctx.seen if key.startswith("device:"))
            else:
                required.add(self._resource_key(template, params))
        stale = sorted(
            key
            for key in required
            if key not in ctx.seen or ctx.seen[key] != current.get(key, 0)
        )
        if stale:
            raise RemoteError(
                ErrorCode.PRECONDITION_FAILED,
                "a resource you depend on was changed in the GUI since you last "
                "saw it; review then retry",
                reason="stale_version",
                data={"stale": stale},
            )

    def _ctrl_resource_versions(self) -> dict[str, int]:
        return dict(self.ctrl.resources_versions())

    def _before_handler(
        self, ctx: SubscriptionCtx, method: str, params: Mapping[str, object]
    ) -> dict[str, int] | None:
        del ctx, params
        if self._agent_policies[method].refresh_after_write:
            return self._ctrl_resource_versions()
        return None

    @staticmethod
    def _self_write_updates(
        seen: Mapping[str, int], before: Mapping[str, int], current: Mapping[str, int]
    ) -> dict[str, int]:
        updates: dict[str, int] = {}
        for key in before.keys() | current.keys():
            old, new = before.get(key, 0), current.get(key, 0)
            if old != new and key in seen and seen[key] == old:
                updates[key] = new
        return updates

    def _owner_success(
        self,
        ctx: SubscriptionCtx,
        method: str,
        params: Mapping[str, object],
        result: Mapping[str, object],
        before: dict[str, int] | None,
    ) -> Callable[[], None] | None:
        assert isinstance(ctx, _ClientCtx)
        policy = self._agent_policies[method]
        current = self._ctrl_resource_versions()
        updates = (
            self._self_write_updates(ctx.seen, before, current)
            if before is not None
            else {}
        )
        if before is not None and policy.created_resource is not None:
            identity = result.get("tab_id")
            if isinstance(identity, str) and identity:
                key = self._resource_key(policy.created_resource, result)
                if before.get(key, 0) == 0 and current.get(key, 0) == 1:
                    updates[key] = 1
        if (
            policy.reveals
            and all(name not in params for name in policy.reveals_without)
            and all(params.get(name) for name in policy.reveals_when_nonempty)
        ):
            for template in policy.reveals:
                key = self._resource_key(template, params)
                updates[key] = current.get(key, 0)
        if not updates:
            return None
        previous = {key: ctx.seen.get(key) for key in updates}
        ctx.seen.update(updates)

        def _rollback() -> None:
            for key, version in updates.items():
                if ctx.seen.get(key) != version:
                    continue
                old = previous[key]
                if old is None:
                    ctx.seen.pop(key, None)
                else:
                    ctx.seen[key] = old

        return _rollback

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
            self._owner_scheduler.post(_run)
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
        reports a fault must not be the faulty channel (ADR-0068)."""
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

    def _on_editor_event(
        self,
        editor_id: str,
        event_name: str,
        payload_factory: Callable[[], object],
    ) -> None:
        """Push a per-editor notification. Runs on the Qt main thread.

        ``event_name`` ∈ {editor_changed, editor_closed}. Only clients that
        subscribed to ``editor_id`` receive it. On editor_closed we also drop the
        id from every client's subscription set. (The editor's resource version
        is bumped at the edit site, not here.)
        """
        closing = event_name == "editor_closed"

        def _predicate(link: ClientLink) -> bool:
            return editor_id in _ctx(link).subscribed_editors

        def _make_line() -> bytes | None:
            try:
                payload = payload_factory()
                if event_name == "editor_changed":
                    from .path_resolver import project_targets

                    body: dict[str, object] = {
                        "paths": project_targets(
                            cast("Iterable[SettableTarget]", payload)
                        )
                    }
                else:
                    body = dict(cast("Mapping[str, object]", payload))
                body["editor_id"] = editor_id
                return encode_line({"event": event_name, "payload": body})
            except Exception:
                logger.exception(
                    "failed to build editor push %s/%s", editor_id, event_name
                )
                return None

        def _on_delivered(link: ClientLink) -> None:
            if closing:
                _ctx(link).subscribed_editors.discard(editor_id)

        self._endpoint.broadcast_lazy(
            _make_line,
            predicate=_predicate,
            on_delivered=_on_delivered,
        )


__all__ = ["ControlOptions", "RemoteControlAdapter"]
