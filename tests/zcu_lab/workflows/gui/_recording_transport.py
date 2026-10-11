"""Shared in-process JSON recording transport for workflow GUI test callers."""

import json
from collections.abc import Callable

from zcu_tools.gui.remote.wire import Request, parse_request
from zcu_tools.mcp.core.bridge import DeliverFn, OnClosedFn

WireRoute = Callable[[Request], dict[str, object]]


class WindowTransport:
    """Move wire messages through the Window route, replacing only TCP.

    route sends one Request through public GUI dispatch and returns its wire reply.
    requests retains submitted identities for forwarding assertions. deliver_reply
    is the bridge callback, or None before attach. is_open is true until close.
    attach supplies bridge callbacks before send_line; close only detaches bytes.
    Owner dispatch and native execution remain with the existing GUI harness.
    """

    def __init__(self, route: WireRoute) -> None:
        """Record requests dispatched to route, which returns ordinary wire replies."""
        self.route = route
        self.requests: list[Request] = []
        self.deliver_reply: DeliverFn | None = None
        self.is_open = True

    def attach(
        self, deliver_reply: DeliverFn, deliver_event: DeliverFn, on_closed: OnClosedFn
    ) -> None:
        """Install reply delivery; event and closed callbacks are unused."""
        self.deliver_reply = deliver_reply

    def send_line(self, payload: dict[str, object]) -> None:
        """Round-trip JSON and dispatch once; raise RuntimeError if closed/unattached.

        JSON encoding, request parsing and owner route failures propagate unchanged.
        """
        if not self.is_open or self.deliver_reply is None:
            raise RuntimeError("Window transport is not attached")
        request = parse_request(json.loads(json.dumps(payload)))
        self.requests.append(request)
        self.deliver_reply(self.route(request))

    def close(self) -> None:
        """Detach the transport without issuing Stop to the Window."""
        self.is_open = False
