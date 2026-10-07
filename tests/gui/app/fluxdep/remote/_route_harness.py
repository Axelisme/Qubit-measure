"""Owner-turn route harness using the shipped Fluxdep adapter and Controller.

Only scheduling is replaced. Request validation, guards, handlers, successful
observations and response encoding all run through the real shared dispatch.
No socket, QApplication, background worker or second seen map is created.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, NotRequired, TypeAlias, TypedDict, TypeVar

from zcu_tools.gui.app.fluxdep.controller import Controller
from zcu_tools.gui.app.fluxdep.remote.service import (
    ControlOptions,
    RemoteControlAdapter,
)
from zcu_tools.gui.app.fluxdep.state import FluxDepState
from zcu_tools.gui.remote.rpc_endpoint import ClientLink
from zcu_tools.gui.remote.wire import Request

_T = TypeVar("_T")


class ErrorReply(TypedDict):
    """Encoded error: code/category, diagnostic message/reason and optional data."""

    code: str
    message: str
    reason: NotRequired[str]
    data: NotRequired[dict[str, object] | None]


class SuccessReply(TypedDict):
    """Successful envelope: id echoes request, ok=True and result is wire data."""

    id: str
    ok: Literal[True]
    result: dict[str, object]


class FailureReply(TypedDict):
    """Failed envelope: id echoes request, ok=False and error explains failure."""

    id: str
    ok: Literal[False]
    error: ErrorReply


RouteReply: TypeAlias = SuccessReply | FailureReply


class ImmediateOwnerScheduler:
    """Run owner-turn callbacks inline; reject foreign-thread calls.

    This scheduler is for synchronous route tests only. Off-owner waits need a
    separately pumped scheduler, not an inline callback on the worker thread.
    """

    def __init__(self) -> None:
        self._owner_id = threading.get_ident()

    def is_owner_thread(self) -> bool:
        """Return whether the caller is this scheduler's constructing thread."""
        return threading.get_ident() == self._owner_id

    def post(self, callback: Callable[[], None]) -> None:
        """Execute callback now on the owner; raise RuntimeError off-owner."""
        if not self.is_owner_thread():
            raise RuntimeError("inline test scheduler requires its owner thread")
        callback()

    def call(self, callback: Callable[[], _T]) -> _T:
        """Return callback's result on the owner; raise RuntimeError off-owner."""
        if not self.is_owner_thread():
            raise RuntimeError("inline test scheduler requires its owner thread")
        return callback()


@dataclass
class RouteHarness:
    """Own a real controller/adapter and per-client transport links.

    ctrl/service expose the shipped seams for native publications and requests.
    links tracks opened connections for cleanup; request_id is a unique counter.
    """

    ctrl: Controller
    service: RemoteControlAdapter
    links: list[ClientLink] = field(default_factory=list)
    request_id: int = 0

    @classmethod
    def create(cls, project_root: str) -> RouteHarness:
        """Build an inert adapter and State rooted at project_root; open no socket."""
        scheduler = ImmediateOwnerScheduler()
        ctrl = Controller(
            FluxDepState(),
            project_root=project_root,
            interactive_owner=scheduler,
        )
        service = RemoteControlAdapter(
            ctrl, ControlOptions(port=0), owner_scheduler=scheduler
        )
        return cls(ctrl, service)

    def client(self) -> ClientLink:
        """Open a fresh unguarded connection with its own real app context."""
        link = ClientLink(f"client-{len(self.links)}", token_required=False)
        self.service.on_client_open(link)
        self.links.append(link)
        return link

    def request(self, link: ClientLink, method: str, **params: object) -> RouteReply:
        """Route a request and decode its envelope; validation errors propagate.

        Shared route raises RemoteError for malformed ParamSpec input before
        dispatch. The transport normally envelopes that error. Owner/guard
        errors are encoded by shared dispatch itself and returned here.
        """
        self.request_id += 1
        request_id = str(self.request_id)
        self.service.route(link, Request(request_id, method, params))
        reply: RouteReply = json.loads(link.outbound.get_nowait())
        assert reply["id"] == request_id
        return reply

    def disconnect(self, link: ClientLink) -> None:
        """Close one owned connection context; reject a link not owned here."""
        self.links.remove(link)
        self.service.on_client_close(link, on_owner_thread=True)

    def close(self) -> None:
        """Close all connection contexts and dispose the idle interactive owner."""
        for link in self.links.copy():
            self.disconnect(link)
        self.ctrl.interactive.dispose()
