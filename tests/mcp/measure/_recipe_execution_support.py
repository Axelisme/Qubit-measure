"""Sample definition and lifetime fixtures for generator execution contracts."""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from threading import Event

from zcu_tools.mcp.measure.recipe import (
    RecipeDefinition,
    RecipeGenerator,
    RecipeInputSchema,
)
from zcu_tools.mcp.measure.recipe_execution import RecipeExecutions
from zcu_tools.mcp.measure.tool_context import MeasureToolContext

from ._recipe_support import LookbackGui


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


class ControlledRunGui(LookbackGui):
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
