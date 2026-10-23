"""Shared GUI-side dispatch, control authority and event subscriptions.

RemoteControlServiceBase routes parsed NDJSON requests to app-provided handlers.
OwnerScheduler serializes State work; bounded readonly waits may run off-main.
Apps inject method registries, wire metadata, event serializers and their optional
control identity owner. Authority checks and commands share one owner turn.

NdjsonRpcEndpoint owns socket lifecycle, framing, authentication, per-client
queues and push delivery. Apps own domain effects. Measure additionally owns
editor lifetime and diagnostics through the existing narrow hooks.
Qt-free and app-free. See ADR-0068 for transport and policy ownership.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from zcu_tools.gui.control_identity import ControlIdentity, ControlIdentityError
from zcu_tools.gui.event_bus import EventMeta, EventOrigin, EventSubscriptions
from zcu_tools.gui.expected_error import ExpectedError
from zcu_tools.gui.remote.control_methods import CONTROL_METHOD_SPECS
from zcu_tools.gui.remote.errors import (
    ErrorCode,
    RemoteError,
    remote_error_from_expected,
)
from zcu_tools.gui.remote.framing import encode_line
from zcu_tools.gui.remote.method_spec import BoundMethod
from zcu_tools.gui.remote.param_spec import validate_params
from zcu_tools.gui.remote.rpc_endpoint import (
    ClientLink,
    ControlOptions,
    NdjsonRpcEndpoint,
)
from zcu_tools.gui.remote.session_discovery import clear_session, write_session
from zcu_tools.gui.remote.wire import Request
from zcu_tools.gui.session.ports import OwnerScheduler

logger = logging.getLogger(__name__)

# An event serializer maps a domain payload to a wire payload (or None to drop).
Serializer = Callable[[Any], Mapping[str, object] | None]


@dataclass(frozen=True, kw_only=True)
class RemoteAppSpec:
    """App-owned dispatch declarations and endpoint identity, shared by adapters.

    Versions and server_name retain the app's wire handshake identity. Registries
    and serializers remain app-owned; this record does not grant control authority
    or select an owner scheduler.
    """

    wire_version: int
    """App wire-protocol version used for handshake compatibility checks."""
    gui_version: int
    """App GUI implementation version advertised by the endpoint."""
    server_name: str
    """App's published server identity in handshake replies."""
    method_registry: Mapping[str, BoundMethod]
    """Wire method names mapped to app handlers and their dispatch policies."""
    event_serializers: Mapping[Any, Serializer]
    """Domain event keys mapped to lazy payload projections; None drops an event."""
    wire_event_name: Callable[[Any], str]
    """Map the same domain event keys to their subscribed wire names."""


def _store_expected_error(
    holder: dict[str, object], exc: ExpectedError, *, origin: str
) -> None:
    """Store generic expected projection, containing projection bugs as controller errors."""
    try:
        error = remote_error_from_expected(exc)
        if isinstance(exc, ControlIdentityError):
            data: dict[str, object] = {
                "holder": exc.status.holder,
                "locked": exc.status.locked,
                "online": exc.status.online,
            }
            error.data = data
        holder["remote_error"] = error
    except Exception as projection_exc:  # projection safety boundary
        logger.exception(
            "%s expected-error projection raised: %s", origin, projection_exc
        )
        holder["controller_error"] = projection_exc


class SubscriptionCtx:
    """Per-connection semantic state attached to ``link.app_ctx``.

    ``client_id`` is a stable opaque connection identity; ``subscribed`` contains
    wire event names. Subclasses add app-owned resources,
    such as editor session ids, by declaring additional ``__slots__``.
    """

    __slots__ = ("client_id", "subscribed")

    def __init__(self) -> None:
        self.client_id = uuid4().hex
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
        owner_scheduler: OwnerScheduler,
        app: RemoteAppSpec,
        control_identity: ControlIdentity | None = None,
    ) -> None:
        """Bind app dispatch and optional owner-thread control authority.

        controller exposes the app bus and commands used by app.method_registry.
        opts configures the socket; owner_scheduler serializes State work.
        app supplies endpoint identity, dispatch methods and event projections.
        control_identity is the application's optional write-authority owner;
        omit it for observe-only apps. When supplied, unmarked methods require
        the calling socket's holder identity, and off-main methods must declare
        readonly=True. Control initialization and transitions share owner turns.
        Construction opens no socket; start() begins serving requests.
        """
        if control_identity is not None:
            for method, entry in app.method_registry.items():
                if entry.off_main_thread and not entry.readonly:
                    raise ValueError(
                        f"off-owner control method must be readonly: {method}"
                    )
        self.control_identity = control_identity
        self.ctrl = controller
        self._opts = opts
        self._wire_version = app.wire_version
        self._method_registry = app.method_registry
        self._event_serializers = app.event_serializers
        self._wire_event_name = app.wire_event_name
        self._owner_scheduler = owner_scheduler
        self._endpoint = NdjsonRpcEndpoint(
            opts,
            wire_version=app.wire_version,
            gui_version=app.gui_version,
            server_name=app.server_name,
            router=self,
        )
        # EventBus subscriptions registered in start(); unsubscribed in stop().
        self._bus: Any = None
        self._bus_subs = EventSubscriptions()

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
        self, ctx: SubscriptionCtx, *, on_owner_thread: bool
    ) -> None:
        """Reclaim extra per-connection resources on drop (e.g. editors). Default: none."""
        del ctx, on_owner_thread

    def _on_client_count_changed(self) -> None:
        """Called on the State owner thread whenever a client changes.

        Override to react to the live-client count changing (e.g. to refresh
        a widget gate). Default: no-op.
        """

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
        endpoint_started = False
        self._subscribe_event_bus()
        try:
            self._extra_start()
            port = self._endpoint.start()
            endpoint_started = True
            self._advertise_session(port)
        except Exception:
            try:
                if endpoint_started:
                    self._endpoint.stop()
            finally:
                try:
                    self._extra_stop()
                finally:
                    self._unsubscribe_event_bus()
            raise
        return port

    def stop(self) -> None:
        """Unwire listeners, then stop the endpoint. Idempotent. Owner thread."""
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

    def has_live_client(self) -> bool:
        """Return True if at least one control client is currently connected.

        Delegates to the endpoint; safe to call from any thread.
        """
        return self._endpoint.has_live_client()

    def on_client_open(self, link: ClientLink) -> None:
        link.app_ctx = self._new_client_ctx()
        # Marshal a count-change notification onto the State owner thread.
        self._owner_scheduler.post(self._on_client_count_changed)

    def on_client_close(self, link: ClientLink, *, on_owner_thread: bool) -> None:
        ctx = _ctx(link)
        self._on_client_close_extra(ctx, on_owner_thread=on_owner_thread)
        control = self.control_identity
        if control is not None:
            if on_owner_thread:
                control.disconnect(ctx.client_id)
            else:
                self._owner_scheduler.post(lambda: control.disconnect(ctx.client_id))
        # Notify the owner on both IO-thread drops and owner-thread stops.
        if on_owner_thread:
            self._on_client_count_changed()
        else:
            self._owner_scheduler.post(self._on_client_count_changed)

    def route(self, link: ClientLink, request: object) -> None:
        """Handle one parsed, authenticated request on the IO thread.

        Owner-thread handlers taking more than 500 ms emit one warning with the
        method and elapsed milliseconds, on success or failure. Only the handler
        call is timed, excluding owner queueing, admission and reply processing.
        Off-main handlers and declared request timeouts retain their policies.
        """
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
            subscribed = self._endpoint.client_state_transaction(
                link, lambda: sorted(_ctx(link).subscribed)
            )
            self._endpoint.reply_ok(
                link,
                rid=req.id,
                result={
                    "events": sorted(
                        self._wire_event_name(k) for k in self._event_serializers
                    ),
                    "subscribed": subscribed,
                },
            )
            return
        # App-specific state-owning methods (e.g. measure-gui's editor.*).
        if self._route_extra(link, req):
            return
        spec = self._method_registry.get(req.method)
        if self.control_identity is not None and req.method in CONTROL_METHOD_SPECS:
            spec = BoundMethod(
                handler=lambda _adapter, params: self._control_command(
                    _ctx(link), req.method, params
                ),
                spec=CONTROL_METHOD_SPECS[req.method],
            )
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
        # an owner-thread hop.
        if spec.params:
            handler_params = validate_params(spec.params, req.params)
        else:
            handler_params = req.params
        self._dispatch_on_owner(link, req.id, req.method, spec, handler_params)

    def _check_control(self, ctx: SubscriptionCtx, method: str) -> None:
        if (
            self.control_identity is None
            or method in CONTROL_METHOD_SPECS
            or method == "rpc.catalog"
        ):
            return
        if not self._method_registry[method].readonly:
            self.control_identity.require_holder(ctx.client_id)

    def _local_modal_open(self) -> bool:
        """Qt adapters query modal presentation on the owner turn; headless has none."""
        return False

    def _control_command(
        self, ctx: SubscriptionCtx, method: str, params: Mapping[str, object]
    ) -> dict[str, object]:
        control = self.control_identity
        assert control is not None
        if method == "control.connect":
            label, mode = params["label"], params["mode"]
            if not isinstance(label, str) or mode not in ("owner", "readonly"):
                raise RemoteError(ErrorCode.INVALID_PARAMS, "invalid control identity")
            if mode == "owner":
                control.connect(
                    ctx.client_id, label, "owner", local_busy=self._local_modal_open()
                )
            else:
                control.connect(ctx.client_id, label, "readonly")
        elif method == "control.acquire":
            control.acquire(ctx.client_id, local_busy=self._local_modal_open())
        elif method == "control.release":
            control.release(ctx.client_id)
        status = control.status()
        return {
            "holder": status.holder,
            "locked": status.locked,
            "online": status.online,
            "me": control.is_holder(ctx.client_id),
        }

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

        def _subscribe() -> list[str]:
            ctx = _ctx(link)
            for ev in events:
                assert isinstance(ev, str)
                ctx.subscribed.add(ev)
            return sorted(ctx.subscribed)

        subscribed = self._endpoint.client_state_transaction(link, _subscribe)
        self._endpoint.reply_ok(link, rid=rid, result={"subscribed": subscribed})

    def _handle_unsubscribe(self, link: ClientLink, rid: str, params) -> None:
        events = params.get("events")
        if not isinstance(events, list):
            raise RemoteError(
                ErrorCode.INVALID_PARAMS, "'events' must be a list of event names"
            )

        def _unsubscribe() -> list[str]:
            ctx = _ctx(link)
            for ev in events:
                if isinstance(ev, str):
                    ctx.subscribed.discard(ev)
            return sorted(ctx.subscribed)

        subscribed = self._endpoint.client_state_transaction(link, _unsubscribe)
        self._endpoint.reply_ok(link, rid=rid, result={"subscribed": subscribed})

    # ------------------------------------------------------------------
    # Dispatch onto the State owner thread (marshal + off-main + policy seams)
    # ------------------------------------------------------------------

    def _dispatch_on_owner(self, link: ClientLink, rid, method, spec, params) -> None:
        holder: dict[str, object] = {}
        bus = self._get_bus()
        ctx = _ctx(link)
        request_origin = EventOrigin(kind="agent", client_id=ctx.client_id)

        if spec.off_main_thread:
            # Blocking handlers wait on this IO worker, never on the State owner.
            # Registry validation requires readonly methods for controlled apps.
            self._run_off_main(spec, params, bus, request_origin, holder)
        else:
            done = threading.Event()

            def _run() -> None:
                # Check current authority at execution, not at request arrival.
                with bus.origin(request_origin):
                    try:
                        self._check_control(ctx, method)
                        started_at = time.perf_counter()
                        try:
                            result = spec.handler(self, params)
                        finally:
                            elapsed_ms = (time.perf_counter() - started_at) * 1000
                            if elapsed_ms > 500:
                                logger.warning(
                                    "owner handler %s took %.1f ms", method, elapsed_ms
                                )
                        if not isinstance(result, dict):
                            raise TypeError(
                                f"handler {method!r} returned non-dict result"
                            )
                        holder["result"] = result
                    except RemoteError as exc:
                        holder["remote_error"] = exc
                    except ExpectedError as exc:
                        _store_expected_error(holder, exc, origin="handler")
                    except Exception as exc:  # Controller error envelope
                        logger.exception("handler raised: %s", exc)
                        holder["controller_error"] = exc
                    finally:
                        done.set()

            self._owner_scheduler.post(_run)
            if not done.wait(timeout=spec.timeout_seconds):
                self._endpoint.reply_error(
                    link,
                    rid=rid,
                    code=ErrorCode.TIMEOUT,
                    message=f"handler did not complete within {spec.timeout_seconds}s",
                )
                return
        self._reply_dispatch(link, rid, (method, params), ctx, holder)

    def _run_off_main(self, spec, params, bus, request_origin, holder) -> None:
        try:
            with bus.origin(request_origin):
                holder["result"] = spec.handler(self, params)
        except RemoteError as exc:
            holder["remote_error"] = exc
        except ExpectedError as exc:
            _store_expected_error(holder, exc, origin="off-main handler")
        except Exception as exc:  # Controller error envelope
            logger.exception("off-main handler raised: %s", exc)
            holder["controller_error"] = exc

    def _reply_dispatch(self, link, rid, request, ctx, holder) -> None:
        method, params = request
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
        self._after_success(ctx, method, params, result)
        self._endpoint.reply_ok(link, rid=rid, result=result)

    # ------------------------------------------------------------------
    # EventBus integration (subscribe on owner thread; push via broadcast)
    # ------------------------------------------------------------------

    def _subscribe_event_bus(self) -> None:
        """Subscribe one callback per serialised event key on the owner thread."""
        bus = self._get_bus()
        self._bus = bus
        subscribed_keys: list[Any] = []
        for key in self._event_serializers:
            cb = self._make_bus_callback(key)
            try:
                self._bus_subs.subscribe_with_meta(bus, key, cb)
            except Exception:  # pragma: no cover — bus.subscribe is straightforward
                logger.exception("Failed to subscribe %s on EventBus", key)
                self._bus_subs.unsubscribe_all()
                raise
            subscribed_keys.append(key)
        logger.debug(
            "event-flow: subscribed %d EventBus events for push: %s",
            len(subscribed_keys),
            [self._wire_event_name(k) for k in subscribed_keys],
        )

    def _unsubscribe_event_bus(self) -> None:
        if self._bus is None:
            return
        self._bus_subs.unsubscribe_all()
        self._bus = None

    def _make_bus_callback(self, key: Any) -> Callable[[Any, EventMeta], None]:
        serializer = self._event_serializers[key]
        wire_name = self._wire_event_name(key)

        def _on_event(payload: Any, meta: EventMeta) -> None:
            # Runs on the State owner thread. The endpoint first selects recipients,
            # then calls this factory on this same thread and revalidates before
            # enqueue.
            def _make_line() -> bytes | None:
                try:
                    wire_payload = serializer(payload)
                except Exception:  # pragma: no cover — serializer must not raise
                    logger.exception("Event serializer for %s raised", wire_name)
                    return None
                if wire_payload is None:
                    return None
                try:
                    return encode_line(
                        {
                            "event": wire_name,
                            "payload": wire_payload,
                            "seq": meta.seq,
                            "origin": {
                                "kind": meta.origin.kind,
                                "operation_id": meta.origin.operation_id,
                            },
                        }
                    )
                except Exception:
                    logger.exception("Failed to encode push line for %s", wire_name)
                    return None

            self._endpoint.broadcast_lazy(
                _make_line,
                predicate=lambda link: wire_name in _ctx(link).subscribed,
            )

        return _on_event


__all__ = [
    "ControlOptions",
    "RemoteAppSpec",
    "RemoteControlServiceBase",
    "SubscriptionCtx",
]
