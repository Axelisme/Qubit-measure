"""Production recipe composition over the shared cooperative async host."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from zcu_tools.experiment.script_host import ScriptHost
from zcu_tools.mcp.measure.recipe_api import RecipeSession
from zcu_tools.mcp.measure.recipe_runtime import RecipeRuntime
from zcu_tools.mcp.measure.tool_context import MeasureToolContext


@dataclass(frozen=True)
class RecipeHost:
    """One execution's scheduler, injected author session and GUI runtime.

    scheduler runs on the caller's worker. session is passed to the authored
    coroutine. runtime owns RPC decoding, cooperative operation drain, runs and
    explicit previews. Hosts are constructed only after inputs validate.
    """

    scheduler: ScriptHost
    session: RecipeSession
    runtime: RecipeRuntime


type RecipeHostFactory = Callable[[MeasureToolContext, Callable[[], None]], RecipeHost]
"""Bind one execution on its worker; the second argument notifies progress changes."""


def create_recipe_host(
    tools: MeasureToolContext, on_change: Callable[[], None]
) -> RecipeHost:
    """Bind the current GUI once and compose its runtime with the production host.

    This can perform the initial connection. Source/transport errors propagate;
    the bound runtime never reconnects or retries an uncertain mutation.
    """
    scheduler = ScriptHost(on_change=on_change)
    runtime = RecipeRuntime(tools, scheduler, on_change=on_change)
    return RecipeHost(scheduler, RecipeSession(runtime), runtime)
