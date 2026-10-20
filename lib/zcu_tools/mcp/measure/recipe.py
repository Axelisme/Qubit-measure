"""Author-facing generator recipe contracts; GUI policy stays in the GUI owner.

Recipes receive a RecipeSession from the driver, not a transport or cfg publication.
Run and analysis use typed yield-from helpers; ask marks a writeback handoff.
The driver stores each named result on its operation, or throws a failure back
into the helper call. Session close closes the generator, so ordinary Python
finally blocks still run.

Cfg preparation uses one fixed connection and the GUI's returned publications.
The injected definitions drive these handles through the shared execution worker.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from threading import Condition, Event

from zcu_tools.mcp.measure.analysis_execution import ExecutionSnapshot
from zcu_tools.mcp.measure.execution_reply import (
    SummaryElements,
    SummaryEstimate,
    SummaryEstimateRows,
    SummaryParameter,
)
from zcu_tools.mcp.measure.recipe_capture import SweepSources as SweepSources
from zcu_tools.mcp.measure.recipe_contracts import (
    AnalysisStage as AnalysisStage,
)
from zcu_tools.mcp.measure.recipe_contracts import (
    MissingParameter as MissingParameter,
)
from zcu_tools.mcp.measure.recipe_contracts import (
    RecipeAnalysis as RecipeAnalysis,
)
from zcu_tools.mcp.measure.recipe_contracts import (
    RecipeDeviceState as RecipeDeviceState,
)
from zcu_tools.mcp.measure.recipe_contracts import (
    RecipeExpression as RecipeExpression,
)
from zcu_tools.mcp.measure.recipe_contracts import (
    RecipeFieldEdit as RecipeFieldEdit,
)
from zcu_tools.mcp.measure.recipe_contracts import (
    RecipeInputSchema as RecipeInputSchema,
)
from zcu_tools.mcp.measure.recipe_contracts import (
    RecipeInputValue as RecipeInputValue,
)
from zcu_tools.mcp.measure.recipe_contracts import (
    RecipeNeedsParameters as RecipeNeedsParameters,
)
from zcu_tools.mcp.measure.recipe_contracts import (
    RecipeParameter as RecipeParameter,
)
from zcu_tools.mcp.measure.recipe_contracts import (
    RecipePropertySchema as RecipePropertySchema,
)
from zcu_tools.mcp.measure.recipe_contracts import (
    RecipeRunSnapshot as RecipeRunSnapshot,
)
from zcu_tools.mcp.measure.recipe_contracts import (
    RecipeScalar as RecipeScalar,
)
from zcu_tools.mcp.measure.recipe_contracts import (
    RecipeWritebackPreview as RecipeWritebackPreview,
)
from zcu_tools.mcp.measure.recipe_contracts import (
    StepStatus as StepStatus,
)
from zcu_tools.mcp.measure.recipe_contracts import (
    WritebackDecision as WritebackDecision,
)
from zcu_tools.mcp.measure.recipe_contracts import (
    WritebackError as WritebackError,
)
from zcu_tools.mcp.measure.recipe_contracts import (
    WritebackFailure as WritebackFailure,
)
from zcu_tools.mcp.measure.recipe_contracts import (
    WritebackItemReceipt as WritebackItemReceipt,
)
from zcu_tools.mcp.measure.recipe_contracts import (
    WritebackReceipt as WritebackReceipt,
)
from zcu_tools.mcp.measure.recipe_contracts import (
    WritebackStageReceipt as WritebackStageReceipt,
)
from zcu_tools.mcp.measure.recipe_contracts import recipe_gui_object
from zcu_tools.mcp.measure.recipe_form import RecipeFieldPath as RecipeFieldPath
from zcu_tools.mcp.measure.recipe_form import RecipeFormField as RecipeFormField
from zcu_tools.mcp.measure.recipe_form import RecipeModuleValues as RecipeModuleValues
from zcu_tools.mcp.measure.recipe_handles import (
    AnalysisStepResult as AnalysisStepResult,
)
from zcu_tools.mcp.measure.recipe_handles import (
    AnalyzeOperation as AnalyzeOperation,
)
from zcu_tools.mcp.measure.recipe_handles import (
    RecipeGenerator as RecipeGenerator,
)
from zcu_tools.mcp.measure.recipe_handles import (
    RecipeOperation as RecipeOperation,
)
from zcu_tools.mcp.measure.recipe_handles import (
    RecipeRun as RecipeRun,
)
from zcu_tools.mcp.measure.recipe_handles import (
    RecipeStep as RecipeStep,
)
from zcu_tools.mcp.measure.recipe_handles import RecipeTabBinding
from zcu_tools.mcp.measure.recipe_handles import (
    RunOperation as RunOperation,
)
from zcu_tools.mcp.measure.recipe_handles import (
    RunStepResult as RunStepResult,
)
from zcu_tools.mcp.measure.recipe_handles import (
    WritebackQuestion as WritebackQuestion,
)
from zcu_tools.mcp.measure.recipe_handles import (
    WritebackResult as WritebackResult,
)
from zcu_tools.mcp.measure.recipe_handles import (
    ask as ask,
)
from zcu_tools.mcp.measure.recipe_tab import RecipeTab as RecipeTab
from zcu_tools.mcp.measure.session import GuiRpcError
from zcu_tools.mcp.measure.tool_context import MeasureToolContext


def _alias_overrides(value: object) -> dict[str, str] | None:
    """Own optional named overrides; reject malformed or empty names before RPC."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("aliases must map non-empty names to components")
    overrides: dict[str, str] = {}
    for alias, component in value.items():
        if (
            not isinstance(alias, str)
            or not alias.strip()
            or not isinstance(component, str)
            or not component.strip()
        ):
            raise ValueError("aliases must map non-empty names to components")
        overrides[alias] = component
    return overrides or None


class RecipeSession:
    """One execution's fixed GUI binding, created and passed by the driver.

    tools is the owning session's tool context. Construction captures its current
    GUI connection (or performs the initial attach); subsequent helpers never
    reconnect, retry stale requests or synthesize GUI defaults. Fast edits remain
    available after cancellation; the recipe decides its policy.
    """

    def __init__(
        self,
        tools: MeasureToolContext,
        *,
        summary_parameters: tuple[SummaryParameter | SummaryElements, ...] = (),
        closed: Event | None = None,
        condition: Condition | None = None,
        consume_cancel: Callable[[], bool] | None = None,
    ) -> None:
        """Bind tools and optional execution lifetime/admission collaborators.

        summary_parameters is the recipe definition's declared cfg fields and
        units for Run capture. Empty tuple records no actual conditions.
        closed/condition are the driver's close event and wake condition; omission
        creates local ones. consume_cancel atomically consumes one pending cancel
        before a Run/analysis start and returns whether to suppress it. This
        callback runs under the RPC lock: it must not block or send RPCs. None
        means no pending cancel. Cfg helpers and raw saves do not consume it.
        """
        self._summary_parameters = summary_parameters
        self._tools = tools.bound()
        self._closed = closed if closed is not None else Event()
        self._condition = condition if condition is not None else Condition()
        self._consume_cancel = consume_cancel
        self._tab: str | None = None
        self._run: RunOperation | None = None
        self._analysis: AnalyzeOperation | None = None
        self._latest_operation: RunOperation | AnalyzeOperation | None = None
        self._written: list[WritebackReceipt] = []

    def _observe_run(self, operation: RunOperation) -> None:
        with self._condition:
            self._run = operation
            self._latest_operation = operation
            self._condition.notify_all()

    def _observe_analysis(self, operation: AnalyzeOperation) -> None:
        with self._condition:
            self._analysis = operation
            self._latest_operation = operation
            self._condition.notify_all()

    def _observe_writeback(self, receipt: WritebackReceipt) -> None:
        with self._condition:
            self._written.append(deepcopy(receipt))
            self._condition.notify_all()

    def pending_operation(self) -> RunOperation | AnalyzeOperation | None:
        """Read the latest unfinished Run/analysis handle, even before its yield.

        Include an admitted start awaiting its receipt and interactive handoff.
        Return None before admission, after completion, or for a suppressed start.
        This performs no GUI request or completion. The driver uses the same
        opaque handle for cancel/finish_early during synchronous author calls.
        """
        with self._condition:
            operation = self._latest_operation
        if isinstance(operation, RunOperation):
            capture = operation.snapshot()
            if (
                capture is not None
                and capture.start_status in ("unknown", "running")
                and capture.outcome is None
            ):
                return operation
        elif isinstance(operation, AnalyzeOperation):
            capture = operation.snapshot()
            if capture is not None and capture.status in ("running", "interactive"):
                return operation
        return None

    def writeback_receipts(self) -> tuple[WritebackReceipt, ...]:
        """Read detached actual writes in call order, including failed prefixes.

        Empty means tab.accept has not returned a receipt. Answers and proposals
        are not writes. Reading sends no GUI request and does not refresh guards.
        """
        with self._condition:
            return deepcopy(tuple(self._written))

    def tab_locator(self) -> str | None:
        """Read the latest prepared/requested tab locator without GUI requests.

        A successful creation receipt or explicit reuse records its locator before
        cfg reads/reset/validation can fail. None means neither has occurred.
        A requested reuse locator does not certify existence, adapter or readiness.
        The driver uses this capture when no Run attempt exists yet.
        """
        with self._condition:
            return self._tab

    def analysis_snapshot(self) -> ExecutionSnapshot | None:
        """Read the latest analysis attempt, including a not-yet-yielded failure.

        No GUI request or completion runs here. None means no analysis attempt or
        one suppressed start; unknown means dispatch occurred without a receipt.
        Completed results, artifacts and failures stay with the original owner.
        """
        with self._condition:
            operation = self._analysis
        return operation.snapshot() if operation is not None else None

    def run_snapshot(self) -> RecipeRunSnapshot | None:
        """Read the latest Run attempt for this execution without GUI requests.

        Return detached conditions/start/completion/save facts, including an
        uncertain or rejected start whose handle was not yielded. None means no
        Run was attempted or the latest start was suppressed by pending cancel.
        This framework observation does not complete Run or refresh guards.
        """
        with self._condition:
            operation = self._run
        return operation.snapshot() if operation is not None else None

    def open_tab(
        self,
        adapter: str,
        *,
        aliases: Mapping[str, str] | None = None,
        reuse: str | None = None,
    ) -> RecipeTab:
        """Open adapter's tab, or validate/reset the named reusable tab.

        adapter is a non-empty public GUI adapter ID. aliases maps caller-owned
        alias names to concrete components and is copied into a single Reset.
        None and {} use global aliases at Reset. Every recipe preparation resets
        before edits, including a new tab. Parameter/library reads, edits and Run
        use that fixed capture until the next explicit Reset. The override is not
        written to global aliases or saved input. reuse=None creates a fresh tab;
        otherwise reuse is a non-empty GUI locator. Busy, mismatched adapters and
        invalid locators fail without creating a fallback. GUI/connection
        errors propagate; empty/non-string names raise ValueError before any GUI request. Identifier errors propagate from the GUI context owner.
        """
        if not adapter or not adapter.strip():
            raise ValueError("adapter must be a non-empty GUI adapter ID")
        overrides = _alias_overrides(aliases)
        if reuse is not None and (not reuse or not reuse.strip()):
            raise ValueError("reuse must be a non-empty GUI tab locator")
        if reuse is None:
            payload: dict[str, object] = {"adapter_name": adapter}
            created = self._tools.send_gui_rpc("tab.new", payload)
            tab = created["tab_id"]
            if not isinstance(tab, str) or not tab:
                raise GuiRpcError(
                    "Invalid tab creation reply", reason="incompatible_wire"
                )
            with self._condition:
                self._tab = tab

        else:
            tab = reuse
            with self._condition:
                self._tab = tab
            observed = self._tools.send_gui_rpc("tab.snapshot", {"tab_id": tab})["tabs"]
            if not isinstance(observed, list) or len(observed) != 1:
                raise GuiRpcError("Requested tab was not found", reason="unknown_tab")
            snapshot = recipe_gui_object(observed[0])
            if snapshot["tab_id"] != tab:
                raise GuiRpcError("Requested tab was not found", reason="unknown_tab")
            if snapshot["adapter_name"] != adapter:
                raise GuiRpcError(
                    "Tab has a different experiment", reason="wrong_experiment"
                )
            interaction = recipe_gui_object(snapshot["interaction"])
            if interaction["is_running"] or interaction["is_analyzing"]:
                raise GuiRpcError("Tab is busy", reason="tab_busy")
        reset: dict[str, object] = {"tab_id": tab}
        if overrides:
            reset["aliases"] = overrides
        self._tools.send_gui_rpc("tab.reset_cfg", reset)
        return RecipeTab(
            RecipeTabBinding(
                self._tools,
                tab,
                observe_run=self._observe_run,
                observe_analysis=self._observe_analysis,
                observe_writeback=self._observe_writeback,
                closed=self._closed,
                condition=self._condition,
                consume_cancel=self._consume_cancel,
                summary_parameters=self._summary_parameters,
            )
        )


@dataclass(frozen=True)
class RecipeDefinition:
    """One explicitly injected recipe and its operator-facing declarations.

    name is its unique non-empty MCP tool name; description explains its workflow.
    input_schema is its handwritten schema and must match run's typed keyword
    parameters. run receives RecipeSession first and returns RecipeGenerator.
    adapter_name is the public GUI adapter ID for guide lookup. summary_parameters
    selects captured actual conditions: SummaryParameter selects a static field,
    SummaryElements expands shared leaves over the current complete list at Run.
    summary_estimates selects native scalar estimates, with SummaryEstimateRows
    expanding over captured native row order. Empty summary tuples are valid. The registry validates definitions
    and schemas before any execution; this value does not register on import.
    """

    name: str
    description: str
    input_schema: RecipeInputSchema
    run: Callable[..., RecipeGenerator]
    adapter_name: str = field(kw_only=True)
    summary_parameters: tuple[SummaryParameter | SummaryElements, ...] = field(
        kw_only=True
    )
    summary_estimates: tuple[SummaryEstimate | SummaryEstimateRows, ...] = field(
        kw_only=True
    )
