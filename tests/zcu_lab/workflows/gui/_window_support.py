"""Window RPC call types, recording transport and explicit Start batches."""

import json
from collections.abc import Callable

from zcu_tools.gui.remote.wire import Request, parse_request
from zcu_tools.mcp.core.bridge import DeliverFn, OnClosedFn

from tests.zcu_lab.workflows._sim_inputs import Inputs
from zcu_lab.workflows.gui.remote import FormWrite

RpcRequest = Callable[[str, dict[str, object]], dict[str, object]]


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
        self.route = route
        self.requests: list[Request] = []
        self.deliver_reply: DeliverFn | None = None
        self.is_open = True

    def attach(
        self, deliver_reply: DeliverFn, deliver_event: DeliverFn, on_closed: OnClosedFn
    ) -> None:
        """Install the bridge reply callback; this route publishes no events."""
        self.deliver_reply = deliver_reply

    def send_line(self, payload: dict[str, object]) -> None:
        """Round-trip the original JSON message without rewriting its identity."""
        if not self.is_open or self.deliver_reply is None:
            raise RuntimeError("Window transport is not attached")
        request = parse_request(json.loads(json.dumps(payload)))
        self.requests.append(request)
        self.deliver_reply(self.route(request))

    def close(self) -> None:
        """Detach the transport without issuing Stop to the Window."""
        self.is_open = False


def start_plan_batch(_inputs: Inputs) -> list[FormWrite]:
    """Select the fixture's native modules through captured public library roots.

    Waveform's reference/inline union has no writable discriminator child path;
    reference selections use existing supported leaves without a tree encoder.
    Omit points so Start must use defaults rather than the old GUI draft.
    """
    return [
        {"path": ["flux_dev"], "value": "flux"},
        {"path": ["flux_unit"], "value": "native"},
        {"path": ["fluxes", "kind"], "value": "list"},
        {"path": ["fluxes", "values"], "value": [1.0, 1.1]},
        {"path": ["pi_length_us"], "value": {"expr": "0.06 * 2"}},
        {"path": ["qub_pulse", "kind"], "value": "ref"},
        {"path": ["qub_pulse", "name"], "value": "Q1.pulses.pi"},
        {"path": ["readout", "kind"], "value": "ref"},
        {"path": ["readout", "name"], "value": "Q1.readout.ro"},
    ]
