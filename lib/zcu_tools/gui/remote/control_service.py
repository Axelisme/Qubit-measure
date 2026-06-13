"""RemoteControlServiceBase — shared scaffolding for each app's RemoteControlAdapter.

The app-agnostic skeleton of every GUI app's *second View* (driving adapter,
ADR-0013): the RPC face onto a ``Controller``, peer to the Qt ``MainWindow``.
Pure transport — the socket lifecycle, NDJSON framing, the per-client writer, the
``wire.version`` / ``auth`` handshakes, and the push fan-out primitive — lives one
layer down in :class:`NdjsonRpcEndpoint`. This base owns the *dispatch
scaffolding* that all three apps share:

  - the :class:`EndpointRouter` seam: ``route`` (events.* state-owning handlers,
    then a ``_route_extra`` hook, then METHOD_REGISTRY lookup + ParamSpec
    validation + main-thread dispatch), ``on_client_open`` / ``on_client_close``;
  - ``_dispatch_on_main``: the marshal onto the Qt main thread (via the shared
    :class:`MainThreadDispatcher`, timeout-bounded), composed with the
    ``off_main_thread`` blocking branch and two policy seams (``_guard`` before
    the handler, ``_after_success`` after) — both no-ops by default;
  - EventBus push: subscribe one callback per serialised event key, serialize,
    and broadcast to subscribed clients.

Each app supplies its domain via ``__init__`` (the method registry, the event
serializers + their wire-name accessor, the wire/gui versions, the server name)
and overrides only the narrow policy seams it needs. The read-only apps
(``fluxdep`` / ``dispersive``) override nothing but ``_get_bus``; ``measure-gui``
adds editor sessions, a version guard, off-main handlers and a diagnostic
channel by overriding the seams.

The event key is a payload ``type`` for all three apps: the base never inspects
the key, it only passes it to ``bus.subscribe`` and the injected
``wire_event_name``; each app supplies ``wire_event_name=lambda p: p.EVENT.value``
so the wire name comes from the payload's own domain enum.

Qt-aware (it composes :class:`MainThreadDispatcher`) but app-free: it imports
nothing from ``gui.app``.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Any

from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.remote.framing import encode_line
from zcu_tools.gui.remote.method_spec import BoundMethod
from zcu_tools.gui.remote.param_spec import validate_params
from zcu_tools.gui.remote.rpc_endpoint import (
    ClientLink,
    ControlOptions,
    MainThreadDispatcher,
    NdjsonRpcEndpoint,
)
from zcu_tools.gui.remote.session_discovery import clear_session, write_session
from zcu_tools.gui.remote.wire import Request

logger = logging.getLogger(__name__)

# An event serializer maps a domain payload to a wire payload (or None to drop).
Serializer = Callable[[Any], Mapping[str, object] | None]


class SubscriptionCtx:
    """Per-connection semantic state attached to ``link.app_ctx``.

    The base only needs the set of wire event names this connection subscribed
    to. Subclasses (measure-gui) extend it with extra per-connection resources
    (e.g. CfgEditor session ids) by subclassing and declaring more ``__slots__``.
    """

    __slots__ = ("subscribed",)

    def __init__(self) -> None:
        self.subscribed: set[str] = set()


def _ctx(link: ClientLink) -> SubscriptionCtx:
    ctx = link.app_ctx
    assert isinstance(ctx, SubscriptionCtx)
    return ctx


class RemoteControlServiceBase:
    """Shared scaffolding for an app's NDJSON RPC ``RemoteControlAdapter``.

    Holds the ``Controller`` (command face, reached by handlers via
    ``adapter.ctrl``) and a :class:`NdjsonRpcEndpoint` (transport). Construct
    after the Controller exists; inert until ``start()``.
    """

    # Subclasses narrow this to their concrete Controller type for handler typing.
    ctrl: Any

    def __init__(
        self,
        controller: Any,
        opts: ControlOptions,
        *,
        wire_version: int,
        gui_version: int,
        server_name: str,
        method_registry: Mapping[str, BoundMethod],
        event_serializers: Mapping[Any, Serializer],
        wire_event_name: Callable[[Any], str],
    ) -> None:
        self.ctrl = controller
        self._opts = opts
        self._wire_version = wire_version
        self._method_registry = method_registry
        self._event_serializers = event_serializers
        self._wire_event_name = wire_event_name
        self._dispatcher = MainThreadDispatcher()
        self._endpoint = NdjsonRpcEndpoint(
            opts,
            wire_version=wire_version,
            gui_version=gui_version,
            server_name=server_name,
            router=self,
        )
        # EventBus subscriptions registered in start(); unsubscribed in stop().
        self._bus: Any = None
        self._bus_subs: list[tuple[Any, Callable[[Any], None]]] = []

    # ------------------------------------------------------------------
    # Policy seams (overridable; defaults give the read-only behaviour)
    # ------------------------------------------------------------------

    def _new_client_ctx(self) -> SubscriptionCtx:
        """Mint per-connection state. Override to use a richer ctx subclass."""
        return SubscriptionCtx()

    def _get_bus(self) -> Any:
        """Return the app's EventBus. Default ``ctrl.bus``; override for variants."""
        return self.ctrl.bus

    def _extra_start(self) -> None:
        """Wire extra app-side listeners before the socket opens. Default: none."""

    def _extra_stop(self) -> None:
        """Unwire extra app-side listeners before the endpoint stops. Default: none."""

    def _route_extra(self, link: ClientLink, req: Request) -> bool:
        """Handle extra state-owning methods (e.g. editor.*). Return True if handled."""
        del link, req
        return False

    def _guard(self, params: Mapping[str, object]) -> None:
        """Pre-handler check on the Qt main thread (e.g. version guard). Default: none."""
        del params

    def _after_success(
        self,
        ctx: SubscriptionCtx,
        method: str,
        params: Mapping[str, object],
        result: Mapping[str, object],
    ) -> None:
        """Post-success bookkeeping on the IO thread (e.g. editor lifecycle). Default: none."""
        del ctx, method, params, result

    def _on_client_close_extra(
        self, ctx: SubscriptionCtx, *, on_main_thread: bool
    ) -> None:
        """Reclaim extra per-connection resources on drop (e.g. editors). Default: none."""
        del ctx, on_main_thread

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> int:
        """Hook EventBus (+ any extra listeners), then start the endpoint.

        Returns the bound port. App-side wiring happens before the socket opens
        so no event is missed. Once the (possibly ephemeral-fallback) real port is
        known, advertise it via session discovery so an agent can find this GUI
        without being told the port.
        """
        self._subscribe_event_bus()
        self._extra_start()
        port = self._endpoint.start()
        self._advertise_session(port)
        return port

    def stop(self) -> None:
        """Unwire listeners, then stop the endpoint. Idempotent. Main thread."""
        self._unsubscribe_event_bus()
        self._extra_stop()
        if self._opts.app_slug:
            clear_session(self._opts.app_slug)
        self._endpoint.stop()

    def _advertise_session(self, port: int) -> None:
        """Write the discovery file for this app's running session (best-effort)."""
        slug = self._opts.app_slug
        if not slug:
            return
        write_session(
            slug,
            port,
            pid=os.getpid(),
            host=self._opts.host(),
            wire_version=self._wire_version,
            started=datetime.now(timezone.utc).isoformat(),
        )

    @property
    def port(self) -> int:
        return self._endpoint.port

    # ------------------------------------------------------------------
    # EndpointRouter seam
    # ------------------------------------------------------------------

    def on_client_open(self, link: ClientLink) -> None:
        link.app_ctx = self._new_client_ctx()

    def on_client_close(self, link: ClientLink, *, on_main_thread: bool) -> None:
        self._on_client_close_extra(_ctx(link), on_main_thread=on_main_thread)

    def route(self, link: ClientLink, request: object) -> None:
        """Handle one parsed, authenticated request on the IO thread."""
        assert isinstance(request, Request)
        req = request
        # Subscription methods are state-owning (per-connection subscription
        # set), so handled here, not via dispatch.
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
                    "events": sorted(
                        self._wire_event_name(k) for k in self._event_serializers
                    ),
                    "subscribed": sorted(_ctx(link).subscribed),
                },
            )
            return
        # App-specific state-owning methods (e.g. measure-gui's editor.*).
        if self._route_extra(link, req):
            return
        spec = self._method_registry.get(req.method)
        if spec is None:
            self._endpoint.reply_error(
                link,
                rid=req.id,
                code=ErrorCode.UNKNOWN_METHOD,
                message=f"unknown method: {req.method!r}",
            )
            return
        # Validate params against the method's ParamSpec contract on the IO
        # thread (pure, no Qt) so malformed requests fail fast without consuming
        # a main-thread hop.
        if spec.params:
            handler_params = validate_params(spec.params, req.params)
        else:
            handler_params = req.params
        self._dispatch_on_main(link, req.id, req.method, spec, handler_params)

    # ------------------------------------------------------------------
    # events.* state-owning handlers
    # ------------------------------------------------------------------

    def _handle_subscribe(self, link: ClientLink, rid: str, params) -> None:
        events = params.get("events")
        if not isinstance(events, list):
            raise RemoteError(
                ErrorCode.INVALID_PARAMS, "'events' must be a list of event names"
            )
        whitelist = {self._wire_event_name(k) for k in self._event_serializers}
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

    # ------------------------------------------------------------------
    # Dispatch onto the Qt main thread (marshal + off-main + policy seams)
    # ------------------------------------------------------------------

    def _dispatch_on_main(self, link: ClientLink, rid, method, spec, params) -> None:
        holder: dict[str, object] = {}

        if spec.off_main_thread:
            # Blocking handler (e.g. operation.await): run on THIS IO worker
            # thread, never the main thread — marshalling it onto the main thread
            # would deadlock (it would occupy the event loop that must dispatch
            # the worker signal it awaits). It must only do thread-safe waiting
            # and must not touch the guard / post-success seams, so neither runs.
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
                # Runs on the Qt main thread (where State + VersionTable live), so
                # the guard's compare-and-act is atomic against any other GUI write.
                try:
                    self._guard(params)
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
        # Every handler returns a wire dict; guard the handler-return invariant
        # (the result, not a ParamSpec-validated input — not redundant).
        result = holder["result"]
        assert isinstance(result, dict), f"handler {method!r} returned non-dict result"
        self._after_success(_ctx(link), method, params, result)
        self._endpoint.reply_ok(link, rid=rid, result=result)

    # ------------------------------------------------------------------
    # EventBus integration (subscribe on main thread; push via broadcast)
    # ------------------------------------------------------------------

    def _subscribe_event_bus(self) -> None:
        """Subscribe one callback per serialised event key on the main thread."""
        bus = self._get_bus()
        self._bus = bus
        for key in self._event_serializers:
            cb = self._make_bus_callback(key)
            try:
                bus.subscribe(key, cb)
            except Exception:  # pragma: no cover — bus.subscribe is straightforward
                logger.exception("Failed to subscribe %s on EventBus", key)
                continue
            self._bus_subs.append((key, cb))
        logger.debug(
            "event-flow: subscribed %d EventBus events for push: %s",
            len(self._bus_subs),
            [self._wire_event_name(k) for k, _ in self._bus_subs],
        )

    def _unsubscribe_event_bus(self) -> None:
        if self._bus is None:
            return
        for key, cb in self._bus_subs:
            try:
                self._bus.unsubscribe(key, cb)
            except Exception:  # pragma: no cover
                logger.exception("Failed to unsubscribe %s on EventBus", key)
        self._bus_subs.clear()
        self._bus = None

    def _make_bus_callback(self, key: Any) -> Callable[[Any], None]:
        serializer = self._event_serializers[key]
        wire_name = self._wire_event_name(key)

        def _on_event(payload: Any) -> None:
            # Runs on the Qt main thread. Serialize and push to subscribers; this
            # is the agent's notification face ("what changed"). Resource version
            # bookkeeping happens at the mutation site, not here.
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
            self._endpoint.broadcast(
                line, predicate=lambda link: wire_name in _ctx(link).subscribed
            )

        return _on_event


__all__ = ["ControlOptions", "RemoteControlServiceBase", "SubscriptionCtx"]
