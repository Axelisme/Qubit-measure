"""Sample definition and lifetime fixtures for generator execution contracts."""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from threading import Event, Timer

from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.measure.recipe import (
    RecipeDefinition,
    RecipeGenerator,
    RecipeInputSchema,
)
from zcu_tools.mcp.measure.recipe_execution import (
    TERMINAL_STATUSES,
    RecipeExecution,
    RecipeExecutions,
)
from zcu_tools.mcp.measure.tool_context import MeasureToolContext

from ._recipe_support import ExecutionGui


def definition(
    run: Callable[..., RecipeGenerator], schema: RecipeInputSchema | None = None
) -> RecipeDefinition:
    """Declare one sample generator without production registration."""
    return RecipeDefinition(
        "sample",
        "Run the sample generator",
        schema
        if schema is not None
        else {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        run,
        adapter_name="lookback",
        summary_parameters=(),
        summary_estimates=(),
    )


@contextmanager
def registry(
    tools: MeasureToolContext, *recipes: RecipeDefinition
) -> Generator[RecipeExecutions]:
    """Drain the public lifetime owner before the client's PNG cleanup."""
    executions = RecipeExecutions(Event(), recipes=recipes)
    try:
        yield executions
    finally:
        executions.stop_admission()
        executions.join()


class ControlledRunGui(ExecutionGui):
    """Hold the native Run until GUI/agent supplies a terminal outcome."""

    def __init__(self) -> None:
        super().__init__()
        self.awaited = Event()
        self.settled = Event()
        self.outcome = "finished"
        self.cancel_count = 0

    def __call__(self, method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "operation.await" and params["operation_id"] == 71:
            self.awaited.set()
            return {
                "reason": "completed" if self.settled.is_set() else "timeout",
                "status": self.outcome if self.settled.is_set() else "running",
            }
        if method == "operation.cancel":
            assert params["operation_id"] == 71
            self.cancel_count += 1
            self.outcome = "cancelled"
            self.settled.set()
            return {"status": "cancelling"}
        return super().__call__(method, params)


def wait_for_event(event: Event) -> None:
    """Observe a worker/RPC arrival after starting it or releasing its predecessor."""
    # The caller has arranged this arrival; the deadline only diagnoses a stuck worker.
    assert event.wait(30), "Worker did not reach the arranged handoff"


def handoff(execution: RecipeExecution) -> ToolReply:
    """Await a started recipe's next interactive/question delivery."""
    # Start or the previous stage's release causes this local delivery.
    return execution.wait(30)


def completed(execution: RecipeExecution) -> ToolReply:
    """Return the captured reply once public progress is terminal.

    The caller must arrange every external release needed for completion, including
    interactive stages. Transient handoffs may still be visible while they settle.
    An independent 30s watchdog raises with the last nonterminal status if the
    recipe stalls or reaches an unexpected handoff. It never closes the execution;
    the caller's registry/session lifetime remains responsible for close and drain.
    """
    expired = Event()
    # The caller's arranged releases cause completion; this timer only guards deadlock.
    watchdog = Timer(30, expired.set)
    try:
        watchdog.start()
        status = execution.snapshot().status
        while status not in TERMINAL_STATUSES:
            assert not expired.is_set(), (
                f"Recipe stalled at nonterminal status: {status}"
            )
            # Yield CPU only; neither this return value nor a sample count proves completion.
            expired.wait(0.01)
            status = execution.snapshot().status
        return execution.wait(0)
    finally:
        watchdog.cancel()
        watchdog.join()
