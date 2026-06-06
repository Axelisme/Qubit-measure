"""RemoteControlAdapter — fluxdep-gui's second View (driving adapter).

The RPC face onto the fluxdep ``Controller``, peer to the Qt ``MainWindow``
(ADR-0013). Pure transport — the socket lifecycle, NDJSON framing, the
per-client writer, the ``wire.version`` / ``auth`` handshakes, and the push
fan-out primitive — lives in the shared :class:`NdjsonRpcEndpoint`. This adapter
keeps only fluxdep's *dispatch policy + domain*:

  - the :class:`EndpointRouter` seam: ``route`` (events.* state-owning handlers,
    then METHOD_REGISTRY lookup + ParamSpec validation + main-thread dispatch),
    ``on_client_open`` / ``on_client_close`` (per-connection subscription set);
  - ``_dispatch_on_main``: marshal the handler onto the Qt main thread via the
    shared :class:`MainThreadDispatcher` and wait (timeout-bounded). fluxdep has
    no version guard, no editor sessions, no off-main handlers — a bare marshal;
  - EventBus push: subscribe the fluxdep payload types, serialize each event,
    and push to subscribed clients via ``endpoint.broadcast``.

Threading is owned by the endpoint (server thread + per-client writer); this
adapter's handlers + bus callbacks run on the Qt main thread.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Callable, Optional

from zcu_tools.gui.app.fluxdep.event_bus import EventBus, Payload
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
    # Type-only: importing Controller at runtime would not cycle here, but the
    # string annotation keeps the import graph lean and pyright happy.
    from zcu_tools.gui.app.fluxdep.controller import Controller

from .dispatch import METHOD_REGISTRY
from .events import EVENT_SERIALIZERS, wire_event_name
from .wire_version import GUI_VERSION, WIRE_VERSION

logger = logging.getLogger(__name__)


class _ClientCtx:
    """fluxdep's per-connection semantic state (attached to ``link.app_ctx``).

    Just the set of wire event names this connection subscribed to.
    """

    __slots__ = ("subscribed",)

    def __init__(self) -> None:
        self.subscribed: set[str] = set()


def _ctx(link: ClientLink) -> _ClientCtx:
    ctx = link.app_ctx
    assert isinstance(ctx, _ClientCtx)
    return ctx


class RemoteControlAdapter:
    """Driving adapter: an NDJSON RPC face onto the fluxdep ``Controller``.

    Holds the ``Controller`` (command face) and a :class:`NdjsonRpcEndpoint`
    (transport). Dispatch handlers receive *this adapter*, so they reach commands
    through ``adapter.ctrl.<m>``. Construct after the Controller exists; inert
    until ``start()``.
    """

    def __init__(self, controller: "Controller", opts: ControlOptions) -> None:
        # Public: dispatch handlers reach the command face through ``adapter.ctrl``.
        self.ctrl = controller
        self._dispatcher = MainThreadDispatcher()
        self._endpoint = NdjsonRpcEndpoint(
            opts,
            wire_version=WIRE_VERSION,
            gui_version=GUI_VERSION,
            server_name="FluxDepRemoteServer",
            router=self,
        )
        # EventBus subscriptions registered in start(); unsubscribed in stop().
        self._bus: Optional[EventBus] = None
        self._bus_subs: list[tuple[type[Payload], Callable[[Payload], None]]] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> int:
        """Hook EventBus, then start the endpoint. Returns the bound port."""
        self._subscribe_event_bus()
        return self._endpoint.start()

    def stop(self) -> None:
        """Unsubscribe EventBus, then stop the endpoint. Idempotent. Main thread."""
        self._unsubscribe_event_bus()
        self._endpoint.stop()

    @property
    def port(self) -> int:
        return self._endpoint.port

    # ------------------------------------------------------------------
    # EndpointRouter seam
    # ------------------------------------------------------------------

    def on_client_open(self, link: ClientLink) -> None:
        link.app_ctx = _ClientCtx()

    def on_client_close(self, link: ClientLink) -> None:
        # fluxdep holds no per-connection resources beyond the subscription set
        # (which dies with the link); nothing to reclaim.
        del link

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
                    "events": sorted(wire_event_name(t) for t in EVENT_SERIALIZERS),
                    "subscribed": sorted(_ctx(link).subscribed),
                },
            )
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
    # Dispatch onto the Qt main thread (bare marshal — no guard / no off-main)
    # ------------------------------------------------------------------

    def _dispatch_on_main(self, link: ClientLink, rid, method, spec, params) -> None:
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
        result = holder["result"]
        assert isinstance(result, dict), f"handler {method!r} returned non-dict result"
        self._endpoint.reply_ok(link, rid=rid, result=result)

    # ------------------------------------------------------------------
    # EventBus integration (subscribe on main thread; push via broadcast)
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
            self._endpoint.broadcast(
                line, predicate=lambda link: wire_name in _ctx(link).subscribed
            )

        return _on_event


__all__ = ["ControlOptions", "RemoteControlAdapter"]
