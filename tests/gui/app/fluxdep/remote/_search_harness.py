"""Real threaded RPC/search runtime with an explicitly pumped State owner."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TypeVar

from zcu_tools.gui.app.fluxdep.controller import Controller
from zcu_tools.gui.app.fluxdep.remote.service import (
    ControlOptions,
    RemoteControlAdapter,
)
from zcu_tools.gui.app.fluxdep.search import FluxDepSearchRuntime
from zcu_tools.gui.app.fluxdep.state import FluxDepState
from zcu_tools.gui.remote.rpc_endpoint import ClientLink
from zcu_tools.gui.remote.wire import Request
from zcu_tools.gui.session.adapters.manual_owner_scheduler import ManualOwnerScheduler
from zcu_tools.gui.session.adapters.thread_pool_background import (
    ThreadPoolBackgroundExecutor,
)
from zcu_tools.gui.session.ports import ProgressEvent
from zcu_tools.gui.session.services.progress import ProgressService

from tests.gui.app.fluxdep.remote._route_harness import RouteHarness, RouteReply

_T = TypeVar("_T")


class QueuedProgress:
    """Deliver progress events on the same owner as the native search runtime."""

    def __init__(self, owner: ManualOwnerScheduler) -> None:
        self.owner = owner
        self.receiver: Callable[[ProgressEvent], None] | None = None

    def set_receiver(self, receiver: Callable[[ProgressEvent], None]) -> None:
        """Install the ProgressService's owner-thread receiver."""
        self.receiver = receiver

    def emit(self, event: ProgressEvent) -> None:
        """Queue worker events; require an installed receiver."""
        receiver = self.receiver
        if receiver is None:
            raise RuntimeError("progress receiver is not installed")
        self.owner.post(lambda: receiver(event))


@dataclass(kw_only=True)
class PumpedRouteHarness(RouteHarness):
    """Run real dispatch on IO threads, and pump callbacks on State's owner.

    owner drives both RPC owner turns and search terminal delivery. background
    is the native Qt-free executor; io_pool permits await concurrent with cancel.
    All are retained until close drains delivery and stops the IO workers.
    """

    owner: ManualOwnerScheduler
    background: ThreadPoolBackgroundExecutor
    io_pool: ThreadPoolExecutor

    @classmethod
    def create(
        cls, project_root: str, state: FluxDepState | None = None
    ) -> PumpedRouteHarness:
        """Compose an inert adapter and real search runtime; open no socket."""
        owner = ManualOwnerScheduler()
        background = ThreadPoolBackgroundExecutor(owner, max_pool_workers=1)
        ctrl = Controller(
            state if state is not None else FluxDepState(),
            project_root=project_root,
            interactive_owner=owner,
            search_runtime=FluxDepSearchRuntime(
                background, ProgressService(QueuedProgress(owner))
            ),
        )
        service = RemoteControlAdapter(
            ctrl, ControlOptions(port=0), owner_scheduler=owner
        )
        return cls(
            ctrl,
            service,
            owner=owner,
            background=background,
            io_pool=ThreadPoolExecutor(max_workers=2),
        )

    def request_async(
        self, link: ClientLink, method: str, **params: object
    ) -> Future[RouteReply]:
        """Send from an IO worker; owner must pump before awaiting owner methods.

        Only one outstanding request per link is supported by this test helper.
        Different links allow a waiting caller and cancelling caller concurrently.
        ParamSpec errors propagate from the Future, as from the base harness.
        """
        self.request_id += 1
        request_id = str(self.request_id)

        def invoke() -> RouteReply:
            self.service.route(link, Request(request_id, method, params))
            reply: RouteReply = json.loads(link.outbound.get_nowait())
            assert reply["id"] == request_id
            return reply

        return self.io_pool.submit(invoke)

    def wait(self, future: Future[_T], timeout: float = 5.0) -> _T:
        """Pump owner delivery until a Future completes, or fail after timeout."""
        deadline = time.monotonic() + timeout
        while not future.done():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("RPC/search fixture did not complete")
            self.owner.pump_once(block=True, timeout=min(0.01, remaining))
        return future.result()

    def request(self, link: ClientLink, method: str, **params: object) -> RouteReply:
        """Run an IO request while pumping the native owner-loop delivery."""
        return self.wait(self.request_async(link, method, **params))

    def close(self) -> None:
        """Cancel search, drain native background deliveries, then close links."""
        self.ctrl.search.begin_close()
        deadline = time.monotonic() + 5.0
        while not self.background.quiesce(timeout=0):
            if time.monotonic() >= deadline:
                raise TimeoutError("search background did not drain")
            self.owner.pump_once(block=True, timeout=0.01)
        self.owner.pump_all()
        self.io_pool.shutdown(wait=True)
        super().close()
