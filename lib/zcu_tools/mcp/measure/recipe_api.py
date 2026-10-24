"""Typed async recipe declarations and GUI convenience operations.

The GUI owns Form validation, source capture and operation admission. Work
helpers await the shared worker host; edits, saves and writeback are synchronous.
Only explicit questions expose authored evidence to the responder.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass, field
from inspect import Parameter, iscoroutinefunction, signature
from math import isfinite
from typing import TYPE_CHECKING, Literal, Protocol, get_args, get_type_hints

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
    field_validator,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import PydanticOmit, PydanticUndefined, core_schema

from zcu_tools.analysis.fitting.quality import FitQuality as FitQuality
from zcu_tools.analysis.fitting.quality import QualityIssue as QualityIssue
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


class Keep:
    """Python-only omitted-input marker; use KEEP, never a JSON string or null.

    In a RecipeInputs subclass, annotate optional Form inputs as T | None | Keep
    and default them to KEEP. Omission retains the Form; explicit None clears it.
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source: object, _handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.is_instance_schema(cls)

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _schema: core_schema.CoreSchema, _handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        raise PydanticOmit

    def __deepcopy__(self, _memo: dict[int, object]) -> Keep:
        return self


# A separate omitted marker is necessary because None is a real Form edit.
KEEP = Keep()


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
    | dict[str, YamlValue]
    | BaseModel
    | list[FormValue]
    | tuple[FormValue, ...]
)
"""Native values, named Form models or unevaluated JSON Form input mappings.

Expr carries unevaluated text. Structured input accepts the existing authored
ModuleRef, ModuleInline, SweepInput branches and Log/List integer-point models
through their BaseModel base. These have no equivalent typed input carrier in
the current remote contract. The GUI owner validates their declared shape,
source and path compatibility; arbitrary BaseModel subclasses gain no new
semantics. JSON mappings retain expression tags, structured sweep values and
nested content for owner validation. They are not a shortcut for selecting
module sources. MCP does not hold a live Form object.
Lists and tuples retain their original order. Values use declared field units.
None passed to set clears nullable input; KEEP alone retains it.
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
class FormField:
    """Detached single-turn Form observation, even when the whole cfg is invalid.

    value is the owner's ordinary YamlValue, including structured mappings.
    This read-only output follows recipe_form.RecipeFormField.value; it does
    not infer an authored model. Authors may decode a mapping with TypeAdapter.
    value_status is available, missing or unavailable; available None is a
    genuine null, not a missing-value marker. errors contains directed failures
    at this field, consuming ancestors or descendants. source is the nearest
    captured reference, or None. lock reports a fixed selection,
    fixed value or declaration readonly constraint; it does not follow source.
    fields lists logical direct-child field names. All values are detached.
    Reading does not install defaults, copy refs or validate the entire cfg.
    """

    value: YamlValue
    value_status: Literal["available", "missing", "unavailable"]
    errors: tuple[FormError, ...]
    source: FormSource | None
    lock: bool
    fields: tuple[str, ...]


class FieldEdit(BaseModel):
    """One ordered override: dotted logical path and native Form input value.

    Decimal path segments address original full-list positions. None clears
    nullable input. Overrides do not choose sources or perform conversions.
    Apply each edit with tab.set after experiment values, without deduplication.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    path: str = Field(min_length=1, description="Nonempty dotted logical Form path.")
    value: FormValue

    @field_validator("path")
    @classmethod
    def validate_path(cls, path: str) -> str:
        """Reject malformed dotted paths before recipe execution."""
        from zcu_tools.mcp.measure.recipe_form import dotted_path

        dotted_path(path)
        return path


class RecipeInputs(BaseModel):
    """Shared recipe controls and ordered Form overrides.

    Subclass with experiment fields using T | None | Keep = KEEP. Omission
    retains Form values; null clears/disables nullable Form fields. Unknown
    fields and non-finite numbers fail admission before opening a tab.
    target and reuse_tab_id control Open/Reset rather than editing Form fields:
    omitted/null target keeps global aliases; omitted/null reuse opens a tab.
    overrides defaults to an empty list and does not accept null.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    target: str | None | Keep = Field(
        default_factory=lambda: KEEP,
        description="Concrete target for this Reset; omission/null keeps global aliases.",
    )
    reuse_tab_id: str | None | Keep = Field(
        default_factory=lambda: KEEP,
        description="Idle matching tab to Reset; omission/null opens a new tab.",
    )
    overrides: list[FieldEdit] = Field(
        default_factory=list,
        description="Ordered edits after experiment values; omitted means no edits, null values clear/disable Form fields.",
    )

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: object) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        for info in cls.model_fields.values():
            if Keep not in get_args(info.annotation):
                continue
            if isinstance(info.default, Keep):
                info.default = PydanticUndefined
                info.default_factory = lambda: KEEP
            if info.description is None or "omission" not in info.description:
                info.description = (info.description or "") + (
                    " Omission retains the Form value; null clears/disables nullable input."
                )
        cls.model_rebuild(force=True)


class RecipeReport(BaseModel):
    """Common report facts, without fit acceptance or writeback policy.

    Add experiment results in a subclass. Authors fill these facts explicitly;
    constructing a report does not analyze, save or apply writeback.
    """

    writeback_applied: bool = Field(
        default=False,
        description="True only when selected writeback items were successfully applied.",
    )
    analysis_failure: str | None = Field(
        default=None,
        description="Owner analysis failure reason; null when no failure is reported.",
    )
    fit_quality: FitQuality | None = Field(
        default=None,
        description="Owner fit diagnostics unchanged; null when no quality result is supplied.",
    )
    diagnostics: tuple[QualityIssue, ...] = Field(
        default=(),
        description="Owner diagnostic paths and reasons in original order, without acceptance thresholds.",
    )


def report_number(value: float | None) -> float | None:
    """Retain a finite scalar unchanged; return None for None, NaN or infinity.

    No rounding, unit conversion, fit selection or writeback decision occurs.
    """
    return value if value is not None and isfinite(value) else None


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
    and reports retains the complete owner report from the single Save RPC.
    A transport failure propagates without claiming an unreceived report. Reserved
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

    def read(self, source_path: str, *, missing_ok: bool = False) -> YamlValue:
        """Read one registered value.read key, such as device.flux.value.

        With missing_ok=True, an unregistered source returns None. Otherwise
        unregistered sources raise GuiRpcError. Other source/transport errors
        always propagate. No tab needs to be open. This reads the session-global
        source, not a tab's fixed Reset capture. It never refreshes the separate
        sources observed by tab.read.
        """
        return self._runtime.read_source(source_path, missing_ok=missing_ok)


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
        target: Mapping[str, str | None | Keep] | None = None,
        reuse: str | None | Keep = None,
    ) -> RecipeTab:
        """Open adapter or validate reuse, then Reset its fixed Form capture.

        target maps alias names to concrete targets, for example {"qubit": "Q1"}.
        Omitted/KEEP/null entries retain global aliases for this Reset.
        reuse is an idle matching tab ID; KEEP/null opens a new tab.
        No automatic tab search or close occurs. Busy, mismatch, source and
        holder failures propagate without retry or fallback.
        """
        return self._runtime.open_tab(
            adapter,
            aliases={
                key: value
                for key, value in (target or {}).items()
                if value is not None and not isinstance(value, Keep)
            },
            reuse=None if isinstance(reuse, Keep) else reuse,
        )

    def device_unit(self, name: str) -> str:
        """Read a named device's cached flux value unit without hardware access.

        Return the owner's unit, such as A, V, native or none (missing valid
        cached Info). Unknown devices and owner/transport errors propagate.
        This never polls hardware or refreshes the device cache.
        """
        return self._runtime.device_unit(name)

    async def ask[A: Answer](self, question: Question[A]) -> A:
        """Wait for a validated answer; raise ScriptCancelled on script cancel.

        Only asked evidence is exposed. Invalid answers leave the request pending.
        Asking does not write, supply a timeout, or synthesize acceptance.
        """
        return await self._runtime.host.ask(question)


class RecipeTab:
    """One GUI tab's Form edits and operations, with no calibration policy.

    Writes use nonempty dotted logical paths; decimal segments are original
    full-list indexes. read uses the same dotted paths. All owner
    admission and transport errors propagate without reconnect or retry.
    """

    def __init__(self, runtime: RecipeRuntime, tab: str, adapter: str) -> None:
        """Bind a Reset tab locator and its registered adapter; use session.open_tab."""
        self._runtime = runtime
        self.tab = tab
        self.adapter = adapter

    def read(self, path: str) -> FormField:
        """Observe one dotted logical field without editing sources.

        Decimal segments are original full-list indexes, as in tones.2.gain.
        Invalid local values remain visible as value_status/errors. Unknown or
        unreachable paths raise. Field validity does not prove whole-cfg validity.
        """
        return self._runtime.read_field(self.tab, path)

    def read_parameter(self, path: str, *, missing_ok: bool = False) -> YamlValue:
        """Read one fixed Open/Reset parameter path, for example $qubit.freq.

        Explicit null returns None. Missing leaves raise GuiRpcError with reason
        parameter_missing, unless missing_ok=True. Malformed paths, unknown
        roles, unavailable sources and transport errors always propagate.
        This does not read the session's live context or refresh any capture.
        """
        return self._runtime.read_parameter(self.tab, path, missing_ok=missing_ok)

    def set[V: FormValue](self, path: str, value: FormValue | list[V] | Keep) -> None:
        """Replace one dotted-path field with native input or Expr.

        Decimal segments are original list indexes, for example tones.2.gain.
        KEEP skips mutation. None clears nullable input. Invalid drafts remain editable;
        structural, source and readonly errors propagate. No source selection,
        conversion, defaults or unit inference occurs.
        """
        if not isinstance(value, Keep):
            self._runtime.set_field(self.tab, path, value)

    def select(self, path: str, option: str | None | Keep) -> None:
        """Select an owner Form option at a dotted logical path.

        option is an exact Form label or captured module key. Inline options
        use the owner converter and its same-name/type inheritance policy.
        Use command(path, "to_inline") for an explicit fixed-source copy.
        None disables only nullable selectors. KEEP skips mutation. Unknown or
        ambiguous options, nonnullable null, ref ancestry and readonly
        restrictions raise without changing input. Legal invalid drafts remain.
        """
        if not isinstance(option, Keep):
            self._runtime.select(self.tab, path, option)

    def command(self, path: str, name: Literal["to_inline"]) -> None:
        """Run the owner's to_inline command at a dotted logical selector path.

        Copies a reference from the fixed capture and preserves child keys.
        An inline module succeeds without changing its draft. Missing source,
        unavailable command, ref ancestry and readonly errors propagate without
        mutation. This does not select fresh defaults or refresh the capture.
        """
        self._runtime.command(self.tab, path, name)

    def check_ready(self) -> None:
        """Validate the whole current Form and lowered cfg without mutation.

        Raise RecipeNeedsParameters containing every owner path and reason.
        This does not Run, refresh fixed captures or change the draft. Structural,
        source, transport and author failures propagate without retry.
        """
        self._runtime.check_ready(self.tab)

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
        updates: Mapping[str, YamlValue | Keep] | None = None,
    ) -> AnalysisResult[T]:
        """Analyze this Run and decode the owner summary as the author's T view.

        No automatic Save or preview occurs. Expected fitting failures and
        descendant cancellation return status="failed", result=None and reason;
        the author decides whether to save, retry, skip or stop. Root cancellation
        raises ScriptCancelled after owner drain. Invalid stages, stale sources,
        incompatible summary fields, programming and transport errors propagate.
        updates maps the adapter's analysis parameter names to ordinary values
        in their declared units; no conversion or Form expression evaluation
        occurs. None or an empty mapping keeps the owner's current parameters.
        KEEP entries are omitted before sending updates to the owner; all KEEP
        is equivalent to no updates. Explicit null values reach the owner
        unchanged. Unknown names and incompatible values fail owner validation
        before analysis starts.
        Updates and analysis share one tracked operation, including Post.
        interactive=True enables native PendingAnalysis/tab_interact/done
        handoff before this await resumes. answer only handles explicit questions.
        """
        filtered_updates = (
            None
            if updates is None
            else {
                key: value
                for key, value in updates.items()
                if not isinstance(value, Keep)
            }
        )
        return await self._runtime.analyze(
            self.tab, stage, result, interactive=interactive, updates=filtered_updates
        )

    def preview(self, pane: Literal["run", "analysis", "post"] = "run") -> None:
        """Attach the pane's current PNG as MCP image content, without saving it.

        Missing/superseded content and transport errors propagate. No image is
        implicitly included in the authored report or read by status().
        """
        self._runtime.preview(self.tab, pane)

    def save(self) -> SaveResult:
        """Synchronously save raw data and all completed artifacts once.

        One RPC finishes on the GUI owner; there is no Save operation to await.
        Retain its complete report for this attempt in runs, and propagate
        transport/admission errors. Per-output failures remain in the report.
        Repeated saves are independent receipts, not accumulated success claims.
        """
        return self._runtime.save(self.tab)

    async def ask_writeback(self) -> WritebackAnswer:
        """Ask for validated choices; raise ScriptCancelled on script cancel.

        Reject unknown, changed or duplicate items before author continuation.
        Answering never writes. Call accept_writeback explicitly to write.
        """
        proposal = self._runtime.writeback_proposal(self.tab)
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
    is the authored BaseModel inferred from the function's second parameter.
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
    """Declare async (session, inputs:BaseModel)->Report without running it.

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
