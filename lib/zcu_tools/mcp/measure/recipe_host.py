"""Host binding for script operations, separate from recipe convenience methods."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import Event
from typing import TYPE_CHECKING

from zcu_tools.core.script_protocol import Operation, Signal
from zcu_tools.mcp.measure.recipe_api import RecipeRun, RecipeSession

if TYPE_CHECKING:
    from zcu_tools.mcp.measure.tool_context import MeasureToolContext


def _unbound_wait(operation: Operation[object], cancellation: Event) -> None:
    raise NotImplementedError(
        f"No recipe operation host is bound for {operation.name!r}"
    )


def _unbound_cancel(operation: Operation[object]) -> None:
    raise NotImplementedError(
        f"No recipe cancellation host is bound for {operation.name!r}"
    )


def _unbound_deliver_cancelled(operation: Operation[object]) -> None:
    raise NotImplementedError(
        f"No recipe cancelled delivery owner is bound for {operation.name!r}"
    )


def _no_runs() -> tuple[RecipeRun, ...]:
    return ()


def _unsupported_signal(signal: Signal) -> None:
    raise ValueError(f"Recipe host does not support {signal.kind!r}")


@dataclass(frozen=True)
class RecipeHost:
    """Bind one execution's session and operation lifetime to the shared driver.

    session is injected into the authored generator. wait runs on its sole worker;
    it must populate operation.delivery and drain terminal work before returning,
    including when cancellation is set. cancel may run on the controlling thread
    concurrently with wait and must request cancellation, not fabricate delivery.
    deliver_cancelled runs on the worker for each new Wait after script cancel;
    it fills cancelled delivery without starting work. It is distinct from cancel,
    which only requests cancellation of already active work. runs returns ordered
    acquisition facts, never save receipts. on_signal handles
    non-cancel controls at yield boundaries on the worker, without re-entering it.
    Default callbacks reject unbound operations and controls; runs is empty.
    This seam hides operation ownership and lets synthetic hosts avoid GUI I/O.
    """

    session: RecipeSession
    wait: Callable[[Operation[object], Event], None] = _unbound_wait
    cancel: Callable[[Operation[object]], None] = _unbound_cancel
    deliver_cancelled: Callable[[Operation[object]], None] = _unbound_deliver_cancelled
    runs: Callable[[], tuple[RecipeRun, ...]] = _no_runs
    on_signal: Callable[[Signal], None] = _unsupported_signal


type RecipeHostFactory = Callable[[MeasureToolContext], RecipeHost]
"""Construct one binding after inputs validate, on the execution's worker."""


def stub_recipe_host(_tools: MeasureToolContext) -> RecipeHost:
    """Return the unimplemented convenience session without connecting to a GUI."""
    return RecipeHost(RecipeSession())
