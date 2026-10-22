"""Typed async recipe declarations and GUI convenience operations.

The GUI owns Form validation, source capture and operation admission. Work
helpers await the shared worker host; edits, saves and writeback are synchronous.
Only explicit questions expose authored evidence to the responder.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass, field
from inspect import Parameter, iscoroutinefunction, signature
from typing import TYPE_CHECKING, Literal, Protocol, get_type_hints

from pydantic import BaseModel

from zcu_tools.cfg_model.template_nodes import Expr as Expr
from zcu_tools.core.script_protocol import (
    Answer,
    OperationStatus,
    Question,
)
from zcu_tools.format_version import YamlValue
from zcu_tools.mcp.measure.raw_save import SaveOutput, SaveReport
from zcu_tools.mcp.measure.writeback_recovery import WritebackRecovery

if TYPE_CHECKING:
    from zcu_tools.mcp.measure.recipe_runtime import RecipeRuntime

type FormPath = tuple[str | int, ...]
"""Nonempty logical field tuple; integers are original full-list indexes."""
type FormValue = (
    str
    | int
    | float
    | complex
    | bool
    | None
    | Expr
    | BaseModel
    | list[FormValue]
    | tuple[FormValue, ...]
)
"""Native values or named Form models, never an unstructured object mapping.

Expr carries unevaluated text. Structured input accepts the existing authored
ModuleRef, ModuleInline, SweepInput branches and Log/List integer-point models
through their BaseModel base. These have no equivalent typed input carrier in
the current remote contract. The GUI owner validates their declared shape,
source and path compatibility; arbitrary BaseModel subclasses gain no new
semantics. MCP does not import GUI Form types or hold a live Form object.
Lists and tuples retain their original order. Values use declared field units.
None passed to set/apply retains input rather than clearing it.
"""


@dataclass(frozen=True)
class FormError:
    """One owner-reported logical validation error.

    path identifies the consuming field with original list indexes; () means
    the whole Form. message is the owner's diagnostic, not a synthesized error.
    """

    path: FormPath
    message: str


@dataclass(frozen=True)
class FormSource:
    """Nearest captured reference supplying a field observation.

    key is the actual captured library key, not an alias. namespace distinguishes
    module from waveform references. path is the field's path inside that Spec.
    Reading a source does not refresh it or grant permission to edit through it.
    """

    key: str
    namespace: Literal["module", "waveform"]
    path: FormPath


@dataclass(frozen=True)
class FormSelector:
    """Current generated selector identity, distinct from its evaluated value.

    mode is ref, inline, disabled, missing or unresolved. key is the actual
    library key in ref mode, otherwise None. Missing means absent input;
    unresolved means a draft without a usable selection. A ref does not imply
    declaration-level readonly status.
    """

    mode: Literal["ref", "inline", "disabled", "missing", "unresolved"]
    key: str | None


@dataclass(frozen=True)
class FormField:
    """Detached single-turn Form observation, even when the whole cfg is invalid.

    value is the owner's ordinary YamlValue, including structured mappings.
    This read-only output follows recipe_form.RecipeFormField.value; it does
    not infer an authored model. Authors may decode a mapping with TypeAdapter.
    value_status is available, missing or unavailable; available None is a
    genuine null, not a missing-value marker. errors contains directed failures
    at this field, consuming ancestors or descendants. source is the nearest
    captured reference, or None. selector is this field's generated selector,
    or None for ordinary scalar/section fields. lock reports a fixed selection,
    fixed value or declaration readonly constraint; it does not follow source.
    fields lists logical direct-child field names. All values are detached.
    Reading does not install defaults, copy refs or validate the entire cfg.
    """

    value: YamlValue
    value_status: Literal["available", "missing", "unavailable"]
    errors: tuple[FormError, ...]
    source: FormSource | None
    selector: FormSelector | None
    lock: bool
    fields: tuple[str, ...]


class FieldEdit(BaseModel):
    """One author-selected escape-hatch edit for tab.apply.

    path is a nonempty logical Form path with original list indexes. value is
    untagged at this layer: a native value, Expr, ModuleRef, ModuleInline or
    other named Form model. A ref selects its actual key; inline requests an
    explicit copy. None retains GUI input. source labels provenance, normally
    explicit for supplied literals and derived for authored expressions.
    This carrier introduces no mandatory overrides field on recipe inputs.
    """

    path: FormPath
    value: FormValue
    source: str = "explicit"


@dataclass(frozen=True)
class RunResult:
    """Terminal acquisition result, without an automatic save or analysis.

    status is completed or failed. run_id is the owner's acquisition identity,
    or None when none exists; it does not prove persistence. partial_result
    marks cooperative interruption and may retain usable data for Save.
    """

    status: Literal["completed", "failed"]
    run_id: str | None
    partial_result: bool = False


@dataclass(frozen=True)
class AnalysisResult[T]:
    """Analysis success or an expected failure that the author can act on.

    Completed result is TypeAdapter(T).validate_python(summary); incompatible
    fields raise without repair. Failed result is None and reason explains the
    expected fitting failure or descendant cancellation. Programming errors
    still raise. Root cancellation raises ScriptCancelled after owner drain.
    """

    result: T | None
    status: Literal["completed", "failed"] = "completed"
    reason: str | None = None


@dataclass(frozen=True)
class SaveResult:
    """One combined synchronous raw/artifact Save attempt for a captured Run.

    status is completed or failed; run_id is the acquisition identity, including
    all-output failures. outputs retains every reported path/status/diagnostic,
    and reports retains each complete owner report in dispatch order. A later
    RPC failure preserves the confirmed prefix with failed status. Reserved
    outputs are not saved files. Repeated saves never accumulate success claims.
    """

    status: OperationStatus
    run_id: str
    outputs: tuple[SaveOutput, ...]
    reports: tuple[SaveReport, ...] = ()


@dataclass(frozen=True)
class WritebackDestination:
    """Owner's concrete destination path and preview-time display context.

    path is the opaque captured actual_path string, never a logical Form path.
    label is destination_context.active_label at preview time, not the captured
    entry/point identity or permission to write. Neither value is reconstructed.
    """

    path: str
    label: str


@dataclass(frozen=True)
class WritebackItem:
    """One existing candidate available for selection, not an arbitrary new value.

    name is the stable owner target name. destination is the captured location.
    value is the owner's proposed YamlValue in destination units. Module
    mappings remain unchanged; no authored model is inferred. quality is the
    producer's named quality model, or None when none was recorded. It carries
    producer-defined fields without a framework acceptance threshold. stage is
    primary or post. Items never authorize writing to another location.
    """

    name: str
    destination: WritebackDestination
    value: YamlValue
    quality: BaseModel | None
    stage: Literal["primary", "post"]


@dataclass(frozen=True)
class WritebackProposal:
    """Detached candidates from the current completed analysis draft.

    items is the ordered collection of existing candidates, possibly empty.
    Reading a proposal neither asks a question nor writes a destination.
    """

    items: tuple[WritebackItem, ...]


class WritebackAnswer(Answer):
    """Decision about a captured writeback question; answering never writes.

    accepted distinguishes acceptance from skip. items selects unchanged
    candidates from that question; skip requires an empty tuple. The host
    rejects unknown/modified/duplicate choices before resuming the author.
    Script cancellation returns None, never a fabricated decision model.
    """

    accepted: bool
    items: tuple[WritebackItem, ...]


class WritebackQuestion(Question[WritebackAnswer]):
    """Ask whether to accept selected existing candidates from proposal.

    proposal contains the destinations, values and recorded quality to show.
    Construct with WritebackQuestion(proposal=proposal). Authors may filter
    proposal.items before asking. The answer tool returns WritebackAnswer.
    Neither the question nor its answer performs a write.
    """

    proposal: WritebackProposal

    def validate_answer(self, answer: Answer) -> None:
        """Reject duplicate, unknown or changed candidates before author delivery."""
        if not isinstance(answer, WritebackAnswer):
            raise TypeError("Writeback requires WritebackAnswer")
        if not answer.accepted and answer.items:
            raise ValueError("Rejected writeback cannot select items")
        if len({item.name for item in answer.items}) != len(answer.items):
            raise ValueError("Duplicate writeback candidate")
        if any(item not in self.proposal.items for item in answer.items):
            raise ValueError("Unknown or modified writeback candidate")


@dataclass(frozen=True)
class WritebackReceipt:
    """GUI-confirmed writeback result, including a successful prefix on failure.

    status is completed, cancelled, failed or unknown after a write timeout.
    submission_id/recovery query the GUI-lifetime owner receipt before resending.
    items lists original captured
    candidates for owner-confirmed applied IDs, in owner order. Actual written
    values and destinations come from the owner's current draft; items is not a
    written-value snapshot. error describes a failed or uncertain attempt,
    otherwise None. No rollback or automatic retry follows a failed write.
    """

    status: OperationStatus | Literal["unknown"]
    items: tuple[WritebackItem, ...]
    error: str | None = None
    submission_id: str | None = None
    recovery: WritebackRecovery | None = None

    def raise_for_error(self) -> None:
        """Raise RuntimeError with the owner diagnostic unless status is completed.

        The receipt retains its confirmed prefix. Authors may inspect it instead
        of raising; accept_writeback always returns the receipt on write failure.
        """
        if self.status != "completed":
            raise RuntimeError(self.error or f"Writeback {self.status}")


@dataclass(frozen=True)
class RecipeRun:
    """One acquisition and its Save attempts in execution order.

    tab is its GUI locator; adapter is the registered name. run_id is the
    acquisition identity, or None before a known owner identity. status records
    completed/failed operation delivery. partial_result records interruption.
    saves retains every Save report attempt, including confirmed prefixes on
    failure. No identity or completed status implies persistence.
    """

    tab: str
    adapter: str
    run_id: str | None
    status: Literal["completed", "failed"] = "completed"
    partial_result: bool = False
    saves: tuple[SaveResult, ...] = ()


@dataclass(frozen=True)
class RecipeOutcome[R: BaseModel]:
    """Engine-owned execution envelope; authors return only their report model.

    status is completed, cancelled or failed, determined by coroutine completion
    and host outcome, not by the report. report is the author's returned R, or
    None after script cancellation or failure. runs is the ordered
    acquisition history. Analysis content is not implicitly exposed here.
    """

    status: OperationStatus
    report: R | None
    runs: tuple[RecipeRun, ...]


class RecipeContext:
    """Read-only registered session sources, independent of tab Form captures."""

    def __init__(self, runtime: RecipeRuntime) -> None:
        """Bind the execution runtime; authors obtain this from session.context."""
        self._runtime = runtime

    def read(self, source_path: str) -> YamlValue:
        """Read one registered value.read key, such as device.flux.value.

        Source/transport errors propagate. No tab needs to be open. This reads
        the session-global source, not a tab's fixed Reset capture. It never
        refreshes the separate sources observed by tab.read.
        """
        return self._runtime.read_source(source_path)


class RecipeSession:
    """One execution's GUI binding; authors receive this from the host."""

    def __init__(self, runtime: RecipeRuntime) -> None:
        """Bind a runtime owning the connection, async host and acquisition history."""
        self._runtime = runtime

    @property
    def context(self) -> RecipeContext:
        """Return a read-only entry/session source reader."""
        return RecipeContext(self._runtime)

    def open_tab(
        self,
        adapter: str,
        *,
        aliases: Mapping[str, str | None] | None = None,
        reuse: str | None = None,
    ) -> RecipeTab:
        """Open adapter or validate reuse, then Reset its fixed Form capture.

        aliases only overrides this Reset; null entries retain global aliases.
        No automatic tab search or close occurs. Busy/mismatched/source/holder
        failures propagate before edits, without retry or fallback.
        """
        return self._runtime.open_tab(adapter, aliases=aliases, reuse=reuse)

    async def ask[A: Answer](self, question: Question[A]) -> A:
        """Wait for a validated answer; raise ScriptCancelled on script cancel.

        Only asked evidence is exposed. Invalid answers leave the request pending.
        Asking does not write, supply a timeout, or synthesize acceptance.
        """
        return await self._runtime.host.ask(question)


class RecipeTab:
    """One GUI tab's Form edits and operations, with no calibration policy.

    Form paths are nonempty logical tuples. Integers are original full-list
    indexes. Strings are literal field names, never split at dots. All owner
    admission and transport errors propagate without reconnect or retry.
    """

    def __init__(self, runtime: RecipeRuntime, tab: str, adapter: str) -> None:
        """Bind a Reset tab locator and its registered adapter; use session.open_tab."""
        self._runtime = runtime
        self.tab = tab
        self.adapter = adapter

    def read(self, path: FormPath) -> FormField:
        """Observe one logical field in one owner turn, without editing sources.

        Invalid local values remain visible as value_status/errors. Unknown or
        unreachable paths raise. Field validity does not prove whole-cfg validity.
        """
        return self._runtime.read_field(self.tab, path)

    def set(self, path: FormPath, value: FormValue) -> None:
        """Set native input or Expr at path; None retains the current GUI input.

        Structured models replace complete branches. Invalid draft errors
        propagate after the owner commits them. No defaults or units are inferred.
        """
        self._runtime.set_field(self.tab, path, value)

    def select(
        self,
        path: FormPath,
        *,
        ref: str | None = None,
        inline: bool = False,
    ) -> None:
        """Select an actual library key or copy the captured selector inline.

        ref and inline=True are mutually exclusive. Neither retains input.
        Owner checks family, readonly and sources. Existing inline is a guarded
        no-op. Copy ref ancestors explicitly before editing descendants.
        """
        self._runtime.select(self.tab, path, ref=ref, inline=inline)

    def apply(self, edits: Sequence[FieldEdit]) -> None:
        """Apply edits in order; first error leaves the successful prefix intact.

        None retains input. Empty edits do nothing. Each source label must be
        nonblank; this escape hatch does not bypass owner validation.
        """
        for edit in edits:
            if not edit.source.strip():
                raise ValueError("Edit source must be nonblank")
            self.set(edit.path, edit.value)

    def call[T](self, method: str, params: BaseModel, *, result: type[T]) -> T:
        """Call one named RPC and decode its reply as result, without retries.

        params is a request model; tab-scoped methods receive this tab's locator.
        result is a response model/dataclass class. Validation errors propagate;
        no field repair or default injection occurs at this boundary.
        """
        return self._runtime.call(self.tab, method, params, result=result)

    async def run(self) -> RunResult:
        """Acquire the current cfg and retain its identity, without saving.

        Descendant cancellation returns completed with partial_result=True.
        Root cancellation raises ScriptCancelled without starting a new Run;
        an in-flight Run drains and retains completion facts first. Synchronous
        finally Save remains available. Genuine operation/admission failures propagate.
        """
        return await self._runtime.run(self.tab, self.adapter)

    async def analyze[T](
        self,
        stage: Literal["primary", "post"],
        *,
        result: type[T],
        interactive: bool = False,
    ) -> AnalysisResult[T]:
        """Analyze this Run and decode the owner summary as the author's T view.

        No automatic Save or preview occurs. Expected fitting failures and
        descendant cancellation return status="failed", result=None and reason;
        the author decides whether to save, retry, skip or stop. Root cancellation
        raises ScriptCancelled after owner drain. Invalid stages, stale sources,
        incompatible summary fields, programming and transport errors propagate.
        interactive=True
        explicitly enables GUI interaction handoff before completion.
        """
        return await self._runtime.analyze(
            self.tab, stage, result, interactive=interactive
        )

    def preview(self, pane: Literal["run", "analysis", "post"] = "run") -> None:
        """Attach the pane's current PNG as MCP image content, without saving it.

        Missing/superseded content and transport errors propagate. No image is
        implicitly included in the authored report or read by status().
        """
        self._runtime.preview(self.tab, pane)

    def save(self) -> SaveResult:
        """Synchronously save raw data then all completed artifacts.

        Both RPCs finish on the GUI owner; there is no Save operation to await.
        Merge their complete reports, retain this attempt in runs, and propagate
        transport/admission errors. Per-output failures remain in the report.
        Repeated saves are independent receipts, not accumulated success claims.
        """
        return self._runtime.save(self.tab)

    def writeback_proposal(self) -> WritebackProposal:
        """Capture completed stages' current candidates without asking or writing."""
        return self._runtime.writeback_proposal(self.tab)

    async def ask_writeback(self) -> WritebackAnswer:
        """Ask for validated choices; raise ScriptCancelled on script cancel.

        Reject unknown, changed or duplicate items before author continuation.
        Answering never writes. Call accept_writeback explicitly to write.
        """
        proposal = self.writeback_proposal()
        return await self._runtime.host.ask(WritebackQuestion(proposal=proposal))

    def accept_writeback(self, items: Sequence[WritebackItem]) -> WritebackReceipt:
        """Write selected IDs from current owner drafts, Primary then Post.

        Unknown/duplicate candidates or author edits to captured items raise
        before writing. Owner draft edits are accepted, including retargets.
        Return original captured candidates for confirmed IDs, not written-value
        snapshots. First failure leaves the successful prefix and diagnostic,
        without retry or rollback.
        """
        return self._runtime.accept_writeback(self.tab, items)


@dataclass(frozen=True)
class RecipeDeclaration:
    """An explicit catalog declaration, without import-time registration or I/O.

    name is the tool name; description explains its operations/handoffs. inputs
    is the authored FormModel inferred from the function's second parameter.
    _body preserves its input/report pairing at the reflection boundary.
    """

    name: str
    description: str
    inputs: type[BaseModel]
    _body: Callable[
        [RecipeSession, BaseModel], Coroutine[object, object, BaseModel]
    ] = field(repr=False)

    def coroutine(
        self,
        session: RecipeSession,
        inputs: BaseModel,
    ) -> Coroutine[object, object, BaseModel]:
        """Bind validated inputs without running authored code; host owns execution."""
        return self._body(session, inputs)


class RecipeDecorator(Protocol):
    """Infer separate Inputs and Report models from an async authored function."""

    def __call__[I: BaseModel, R: BaseModel](
        self,
        function: Callable[[RecipeSession, I], Coroutine[object, object, R]],
    ) -> RecipeDeclaration:
        """Capture async (session, inputs:I)->R, rejecting invalid signatures."""
        raise NotImplementedError


def recipe(name: str, description: str) -> RecipeDecorator:
    """Declare async (session, inputs:FormModel)->Report without running it.

    name/description must be nonblank. Return the declaration in an explicit
    RECIPES tuple. Inputs and Report must be named BaseModel subclasses. No
    implicit registry, mandatory overrides or GUI connection is created.
    """
    if not name.strip() or not description.strip():
        raise ValueError("Recipe name and description must be nonblank")
    return _RecipeDecorator(name, description)


@dataclass(frozen=True)
class _RecipeDecorator:
    name: str
    description: str

    def __call__[I: BaseModel, R: BaseModel](
        self,
        function: Callable[[RecipeSession, I], Coroutine[object, object, R]],
    ) -> RecipeDeclaration:
        if not iscoroutinefunction(function):
            raise TypeError("Recipe must be a coroutine function")
        parameters = tuple(signature(function).parameters.values())
        if len(parameters) != 2 or any(
            item.kind
            not in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
            for item in parameters
        ):
            raise TypeError("Recipe must accept exactly two positional parameters")
        hints = get_type_hints(function)
        inputs = hints.get(parameters[1].name)
        if not isinstance(inputs, type) or not issubclass(inputs, BaseModel):
            raise TypeError("Recipe inputs annotation must be a FormModel class")
        returned = hints.get("return")
        if not isinstance(returned, type) or not issubclass(returned, BaseModel):
            raise TypeError("Recipe return annotation must be a Report model")

        input_model: type[I] = hints[parameters[1].name]

        def body(
            session: RecipeSession,
            values: BaseModel,
        ) -> Coroutine[object, object, BaseModel]:
            return function(session, input_model.model_validate(values))

        return RecipeDeclaration(self.name, self.description, inputs, body)
