"""Arrival-driven admission for recording recipe transport tests."""

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from time import monotonic, sleep

from zcu_tools.mcp.core.reply import ToolReply

from ._support import MeasureClient


def wait_for_event(event: Event, reason: str) -> None:
    """Wait for an arranged fake arrival/release; fail with its role on deadlock."""
    # The caller has arranged a producer; this is a deadlock watchdog, not a delay.
    assert event.wait(30), reason


def start_recipe_at(
    client: MeasureClient,
    arrived: Event,
    arguments: Mapping[str, object],
) -> ToolReply:
    """Admit the real pipeline, await its fake RPC arrival, then poll local capture.

    The responder must signal ``arrived`` inside the target RPC. The caller owns
    releasing any blocked responder in finally and closing the session to join
    the real recipe worker, including when an assertion fails.
    """
    execution = client.context.session.recipes.start(
        client.context, "pipeline", arguments
    )
    wait_for_event(arrived, "Recipe worker must reach the arranged RPC")
    return execution.wait(0)


def cancel_while_rpc_held(
    client: MeasureClient, execution: str, release: Event
) -> ToolReply:
    """Latch real cancellation on another thread before releasing an admitted RPC.

    Observe only public captured progress while the control caller waits for RPC
    serialization. Always release the responder and collect caller exceptions.
    The owning fixture closes the session to join the real recipe worker.
    """
    with ThreadPoolExecutor(max_workers=1) as pool:
        request = pool.submit(client.call, "cancel", {"execution": execution})
        try:
            # Cancel must latch before waiting for RPC serialization; guard deadlock.
            deadline = monotonic() + 30
            state = client.call("status", {"execution": execution})
            while not state["cancel_requested"]:
                if request.done():
                    request.result()  # Propagate caller failure instead of timing out.
                    raise AssertionError(f"Cancel returned without latching: {state}")
                assert monotonic() < deadline, f"Cancel did not latch: {state}"
                sleep(0.001)  # Yield CPU; only the captured condition ends the wait.
                state = client.call("status", {"execution": execution})
        finally:
            release.set()
            # Releasing the admitted RPC lets the serialized control caller finish.
            reply = request.result(timeout=30)
        return reply
