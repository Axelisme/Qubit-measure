"""Session-local admission and lookup for injected async recipe declarations."""

from collections.abc import Mapping, Sequence
from threading import Event, Lock

from zcu_tools.mcp.measure.recipe_api import RecipeDeclaration
from zcu_tools.mcp.measure.recipe_host import RecipeHostFactory, create_recipe_host
from zcu_tools.mcp.measure.script_recipe_execution import (
    SCRIPT_TERMINAL,
    ScriptRecipeExecution,
    ScriptRecipeSnapshot,
)
from zcu_tools.mcp.measure.session import GuiRpcError
from zcu_tools.mcp.measure.tool_context import MeasureToolContext


class RecipeExecutions:
    """Admit one active recipe on the owning tools context and retain its history.

    closed belongs to the session. recipes is an explicit declaration sequence;
    duplicate/blank names and invalid input schemas fail before GUI work.
    host_factory optionally replaces the GUI collaborator, not the shared host.
    """

    def __init__(
        self,
        closed: Event,
        *,
        recipes: Sequence[RecipeDeclaration],
        host_factory: RecipeHostFactory | None = None,
    ) -> None:
        self._closed = closed
        self._lock = Lock()
        self._definitions: dict[str, RecipeDeclaration] = {}
        self._host_factory = host_factory or create_recipe_host
        self._executions: dict[str, ScriptRecipeExecution] = {}
        self._tools: MeasureToolContext | None = None
        for definition in recipes:
            if not definition.name.strip() or not definition.description.strip():
                raise ValueError("Recipe requires name and description")
            if definition.name in self._definitions:
                raise ValueError(f"Duplicate recipe name: {definition.name!r}")
            definition.inputs.model_json_schema()
            self._definitions[definition.name] = definition

    @property
    def definitions(self) -> tuple[RecipeDeclaration, ...]:
        """Return declarations in injected order, without registering anything."""
        return tuple(self._definitions.values())

    def start(
        self, tools: MeasureToolContext, recipe: str, arguments: Mapping[str, object]
    ) -> ScriptRecipeExecution:
        """Start one declaration, or reject unknown/busy/closed/replacement contexts.

        Inputs validate on the worker before binding the GUI. Terminal records,
        including needs_parameters and invalid inputs, release admission and
        remain queryable. A nonterminal recipe blocks admission with its ID.
        """
        definition = self._definitions.get(recipe)
        if definition is None:
            raise GuiRpcError(f"Unknown recipe: {recipe!r}", reason="unknown_recipe")
        with self._lock:
            if self._closed.is_set():
                raise GuiRpcError("MCP session is closed", reason="session_closed")
            if self._tools is not None and tools is not self._tools:
                raise GuiRpcError(
                    "Recipe tools are not the owning fixed context",
                    reason="wrong_binding",
                )
            for existing in self._executions.values():
                snapshot = existing.snapshot()
                if snapshot.status not in SCRIPT_TERMINAL:
                    raise GuiRpcError(
                        f"Recipe {snapshot.execution} is {snapshot.status}",
                        reason="recipe_busy",
                    )
            self._tools = tools
            execution = ScriptRecipeExecution(
                tools, definition, arguments, host_factory=self._host_factory
            )
            self._executions[execution.snapshot().execution] = execution
            return execution

    def get(self, execution: str) -> ScriptRecipeExecution:
        """Read a session-local ID; unknown IDs raise unknown_execution."""
        with self._lock:
            found = self._executions.get(execution)
        if found is None:
            raise GuiRpcError(
                f"Unknown execution: {execution!r}", reason="unknown_execution"
            )
        return found

    def for_op(self, op: int) -> ScriptRecipeExecution | None:
        """Find a captured Run/analysis session handle without GUI reads."""
        if isinstance(op, bool) or op <= 0:
            raise ValueError("op must be a positive integer")
        for execution in self._items():
            snapshot = execution.snapshot()
            if (
                snapshot.pending_analysis is not None
                and snapshot.pending_analysis.op == op
            ):
                return execution
            if any(
                detail.op == op
                or any(analysis.op == op for analysis in detail.analyses)
                for detail in snapshot.details
            ):
                return execution
        return None

    def snapshots(self) -> tuple[ScriptRecipeSnapshot, ...]:
        """Read detached local captures in admission order, without GUI requests."""
        return tuple(execution.snapshot() for execution in self._items())

    def stop_admission(self) -> None:
        """Reject new work and cooperatively stop workers; join drains separately."""
        self._closed.set()
        for execution in self._items():
            execution.close()

    def join(self) -> None:
        """Drain stopped workers, without deleting saved artifacts."""
        for execution in self._items():
            execution.join()

    def _items(self) -> tuple[ScriptRecipeExecution, ...]:
        with self._lock:
            return tuple(self._executions.values())
