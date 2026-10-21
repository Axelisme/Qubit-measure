"""Typed recipe declarations and unimplemented convenience contracts.

The decorator retains an authored generator for the script recipe host.
Session and tab convenience methods still raise NotImplementedError. The GUI
owns Form validation, fixed source captures, admission and operation lifetime.
Connection, holder, busy, source and owner errors propagate without reconnect,
retry or rollback. Work helpers use yield from. Form edits and writeback
acceptance are ordinary calls. Only ask exposes a question to the agent.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from inspect import Parameter, isgeneratorfunction, signature
from typing import Literal, Protocol, get_args, get_origin, get_type_hints

from pydantic import BaseModel

from zcu_tools.cfg_model.template_nodes import Expr as Expr
from zcu_tools.core.script_protocol import (
    Answer,
    OperationStatus,
    Question,
    ScriptGenerator,
)
from zcu_tools.mcp.measure.raw_save import SaveOutput

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
type RecipeGenerator[R: BaseModel] = ScriptGenerator[R | None]
"""Recipe body yielding requests and returning a report, or None on cancellation."""


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

    value is an evaluated ordinary leaf or a named structured Form value.
    value_status is available, missing or unavailable; available None is a
    genuine null, not a missing-value marker. errors contains directed failures
    at this field, consuming ancestors or descendants. source is the nearest
    captured reference, or None. selector is this field's generated selector,
    or None for ordinary scalar/section fields. lock reports a fixed selection,
    fixed value or declaration readonly constraint; it does not follow source.
    fields lists logical direct-child field names. All values are detached.
    Reading does not install defaults, copy refs or validate the entire cfg.
    """

    value: FormValue
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

    status is completed, cancelled or failed. run_id is the acquisition identity,
    or None if acquisition never started; it does not prove persistence.
    cancelled is true only for cooperative cancellation. Early finish with
    usable data is completed. A cancelled run may retain partial data for Save.
    """

    status: OperationStatus
    run_id: str | None
    cancelled: bool


class AnalysisResult[T]:
    """Terminal analysis outcome retaining the adapter's native result model T.

    status is completed, cancelled or failed. cancelled is true only for
    cooperative cancellation. result returns T on completed analysis; access
    without a completed result raises RuntimeError with the outcome diagnostic.
    Check cancelled before reading result. Genuine analysis failures propagate
    from analyze, rather than fabricating a model or an empty success value.
    """

    status: OperationStatus
    cancelled: bool

    @property
    def result(self) -> T:
        """Return the adapter result, or raise RuntimeError if none completed.

        No dict projection or scalar extraction occurs here. For example, a
        T1AnalyzeResult retains its typed t1 and t1_err properties.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class SaveResult:
    """One raw-save attempt for the tab's captured acquisition.

    status is completed, cancelled or failed. run_id is the acquisition identity,
    including when every output failed. outputs preserves this attempt's native
    output statuses, paths and diagnostics; an empty tuple proves no success.
    Reserved paths are not saved files. Later saves do not accumulate receipts.
    """

    status: OperationStatus
    run_id: str
    outputs: tuple[SaveOutput, ...]


@dataclass(frozen=True)
class WritebackDestination:
    """Captured destination of an existing proposal, not permission to write.

    entry_id identifies the entry. point_id is its captured saved point, or
    None for entry-level destinations. path is the concrete parameter/module
    field path. label is the owner's human-readable destination label.
    """

    entry_id: str
    point_id: str | None
    path: FormPath
    label: str


@dataclass(frozen=True)
class WritebackItem:
    """One existing candidate available for selection, not an arbitrary new value.

    name is the stable owner target name. destination is the captured location.
    value is the proposed native value in destination units. quality is the
    producer's named quality model, or None when none was recorded. It carries
    producer-defined fields without a framework acceptance threshold. stage is
    primary or post. Items never authorize writing to another location.
    """

    name: str
    destination: WritebackDestination
    value: FormValue
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

    accepted is true when the responder selected candidates for acceptance.
    items contains only selected candidates from the question's proposal, in
    proposal order. cancelled defaults to False and is True only for cooperative
    cancellation, which sets accepted=False and items=(). Rejection (skip) is
    not cancellation: accepted=False, items=() and cancelled=False.
    The host rejects unknown, modified or duplicate candidates before delivery.
    """

    accepted: bool
    items: tuple[WritebackItem, ...]
    cancelled: bool = False


class WritebackQuestion(Question[WritebackAnswer]):
    """Ask whether to accept selected existing candidates from proposal.

    proposal contains the destinations, values and recorded quality to show.
    Construct with WritebackQuestion(proposal=proposal). Authors may filter
    proposal.items before asking. The answer tool returns WritebackAnswer.
    Neither the question nor its answer performs a write.
    """

    proposal: WritebackProposal


@dataclass(frozen=True)
class WritebackReceipt:
    """GUI-confirmed writeback result, including a successful prefix on failure.

    status is completed, cancelled or failed. items lists only confirmed applied
    candidates in owner order. error describes a failed or uncertain attempt,
    otherwise None. No rollback or automatic retry follows a failed write.
    """

    status: OperationStatus
    items: tuple[WritebackItem, ...]
    error: str | None = None


@dataclass(frozen=True)
class RecipeRun:
    """One acquisition in recipe execution order.

    tab is the GUI tab locator, adapter is its catalog name, and run_id is the
    acquisition identity or None if it never started. Identity is not a Save
    receipt. Multi-tab recipes retain every acquisition in this ordered list.
    """

    tab: str
    adapter: str
    run_id: str | None


@dataclass(frozen=True)
class RecipeOutcome[R: BaseModel]:
    """Engine-owned execution envelope; authors return only their report model.

    status is completed, cancelled or failed, determined by generator completion
    and host outcome, not by the report. report is the author's returned R, or
    None after cancellation/failure without a report. runs is the ordered
    acquisition history. Analysis content is not implicitly exposed here.
    """

    status: OperationStatus
    report: R | None
    runs: tuple[RecipeRun, ...]


class RecipeContext:
    """Read-only entry/session source access, separate from Form paths.

    The session owns the fixed source capture. This API does not write values,
    connect devices, refresh capture or copy library content into a tab.
    """

    def read(self, source_path: str) -> FormValue:
        """Read a source scalar such as '$resonator.freq' in its declared units.

        source_path is a nonblank logical parameter path, not arithmetic or a
        Form tuple. Return the captured ordinary value, including None for a
        genuine null or an unset parameter. Unbound aliases, malformed paths,
        absent context and owner/transport errors propagate. This read does not
        prove that any tab cfg is valid and never changes its input.
        """
        raise NotImplementedError


class RecipeSession:
    """One execution's GUI connection, read-only context and typed question API.

    The host injects the session; authors do not construct a connection or
    driver. Tabs share the GUI's current entry/session. The host retains
    operation ownership until completion, cancellation and cleanup finish.
    """

    @property
    def context(self) -> RecipeContext:
        """Return this session's captured source reader, never a write capability."""
        raise NotImplementedError

    def open_tab(
        self,
        adapter: str,
        *,
        aliases: Mapping[str, str | None] | None = None,
        reuse: str | None = None,
    ) -> RecipeTab:
        """Open or explicitly reuse a tab, then Reset its fixed Form capture.

        adapter is a registered catalog name such as 'twotone/t1'. aliases
        overrides alias roles for this Reset only; omitted/empty mappings and
        None values retain GUI aliases. reuse is an idle matching tab ID, or
        None to create a new tab. No automatic tab search or close occurs.
        Inputs are not accepted or copied. Unknown/mismatched/busy tabs and
        source, holder or connection failures propagate before recipe edits.
        """
        raise NotImplementedError

    def ask[A: Answer](self, question: Question[A]) -> ScriptGenerator[A]:
        """Yield Ask(question), wait without a timeout, and return a validated A.

        question is an authored Question[A], including custom named models.
        Only this explicit handoff exposes its evidence to the responder.
        Invalid replies fail before resuming. Asking never writes; cancellation
        follows host control policy and does not synthesize an accepted answer.
        """
        raise NotImplementedError


class RecipeTab:
    """Form operations and acquisition helpers for one session-owned GUI tab.

    Paths are nonempty logical tuples. Strings name fields and nonnegative
    integers name original full-list positions, never participant ordering.
    Dots inside field names are literal. Reads use the fixed Reset capture;
    edits do not refresh it. This API contains no calibration policy.
    """

    def read(self, path: FormPath) -> FormField:
        """Observe one logical Form field without materializing or editing input.

        path is a Form tuple, not a '$source.path' string. Return field-local
        value/status/errors and selector/source metadata in one owner turn.
        Unknown or unreachable paths raise; ordinary invalid values remain
        visible through value_status and errors. Read grants no write access.
        Use session.context.read for entry/session source scalars.
        """
        raise NotImplementedError

    def set(self, path: FormPath, value: FormValue) -> None:
        """Write a native value, Expr or complete named Form model at path.

        null keeps the GUI value. None cannot clear a nullable field; clearing
        uses L1 RPC. Expr text remains authored input, never evaluated in MCP.
        A complete SweepInput replaces its active branch; partial edits address
        active leaves. Owner path, readonly, source and validation errors
        propagate; invalid drafts may remain visible. No default is invented.
        """
        raise NotImplementedError

    def select(
        self,
        path: FormPath,
        *,
        ref: str | None = None,
        inline: bool = False,
    ) -> None:
        """Select an actual reference key or explicitly copy a selector inline.

        ref is an actual library key, or None to retain the GUI selection.
        inline=True copies the captured selector while preserving child refs;
        copy ref ancestors before editing their descendants. Supply either a
        non-null ref or inline=True, never both. With neither, retain input.
        Copying an existing inline selector is a guarded no-op. Unknown keys,
        incompatible families, missing sources and fixed selections raise.
        """
        raise NotImplementedError

    def apply(self, edits: Sequence[FieldEdit]) -> None:
        """Apply authored escape-hatch edits in order, with no batch atomicity.

        edits contains FieldEdit models; each value follows set/select semantics.
        None retains GUI input. Stop on the first error and retain any successful
        prefix. Overrides do not bypass calibration or source requirements.
        Empty edits changes nothing. Provenance uses each edit's source label.
        """
        raise NotImplementedError

    def call[T](self, method: str, params: BaseModel, *, result: type[T]) -> T:
        """Call a GUI RPC with named request and response models, without retry.

        method is a registered RPC name. params supplies named parameters;
        the host supplies this tab's locator for tab-scoped methods. result is the
        response model class, including a dataclass, not a model instance.
        Return that validated native T. For cached device reads use
        'device.snapshot', params with name, and a response model with snapshot.
        This does not connect/poll hardware. RPC admission and response-model
        mismatches propagate; no dict guessing, reconnect or retry occurs.
        """
        raise NotImplementedError

    def run(self) -> ScriptGenerator[RunResult]:
        """Yield Wait for acquisition of the current valid cfg, without saving.

        Return status, run_id and cancelled. Check cancelled before continuing;
        authors may still save partial data. The GUI performs fresh admission
        and captures cfg for this acquisition. Later edits cannot change it.
        Validation/admission and genuine operation failures propagate.
        """
        raise NotImplementedError

    def analyze[T](
        self,
        stage: Literal["primary", "post"],
        *,
        result: type[T],
        interactive: bool = False,
    ) -> ScriptGenerator[AnalysisResult[T]]:
        """Yield Wait for stage analysis of this tab's captured acquisition.

        stage selects primary or post. result is the adapter analysis result
        model class, for example T1AnalyzeResult, not an instance or dict schema.
        It infers T so fit.result.t1 retains its native field type. The host
        rejects incompatible result models instead of projecting a dict.
        interactive=False exposes no analysis evidence to the agent. True
        explicitly hands off the analysis tab for interaction before completion.
        Return a cancellation outcome for cleanup; genuine failures propagate.
        Missing acquisition or required primary analysis fails before dispatch.
        """
        raise NotImplementedError

    def preview(self, pane: Literal["run", "analysis", "post"] = "run") -> None:
        """Attach the selected pane's current PNG to this execution's MCP reply.

        pane selects the Run, Primary analysis or Post analysis pane. Use the
        ordinary call 'tab.preview()' to explicitly expose its current image
        as MCP image content, not a yield-from handoff. Return None; the image
        is not a Report field or a base64 payload returned to the author.
        This reads the current PNG without saving a file or creating an
        identity. Unavailable or superseded Run data raises; owner and
        transport failures propagate. No automatic preview occurs.
        """
        raise NotImplementedError

    def save_raw(self) -> ScriptGenerator[SaveResult]:
        """Yield Wait for a raw-save attempt of this tab's captured acquisition.

        Use 'saved = yield from tab.save_raw()', not a plain call. Return this
        attempt's outputs and run_id. Inspect output statuses for partial or
        complete save failure; an identity or reserved path proves no success.
        Missing acquisition, transport and admission failures propagate. Saving
        neither analyzes nor writes calibration values, and never retries.
        """
        raise NotImplementedError

    def writeback_proposal(self) -> WritebackProposal:
        """Read existing completed-analysis candidates without asking or writing.

        Return ordered destinations, values and recorded quality. An empty
        items tuple means no candidates. Missing/stale analysis draft or owner
        errors propagate. Authors may filter items before explicitly asking.
        """
        raise NotImplementedError

    def ask_writeback(self) -> ScriptGenerator[WritebackAnswer]:
        """Ask about the current proposal and return a WritebackAnswer.

        This is writeback_proposal followed by session.ask of WritebackQuestion.
        Use 'answer = yield from tab.ask_writeback()'. First check
        answer.cancelled and return None from the recipe when True. Otherwise,
        call accept_writeback(answer.items) separately when answer.accepted is
        True, then return the author's Report whether accepted or skipped.
        Asking never writes, substitutes a default decision or waits with a
        timeout. Proposal and question errors propagate.
        """
        raise NotImplementedError

    def accept_writeback(self, items: Sequence[WritebackItem]) -> WritebackReceipt:
        """Apply selected existing candidates to the GUI's current draft.

        items must be unmodified candidates from a proposal; () applies none.
        No arbitrary new value or destination is accepted. The GUI rechecks
        holder, destination and draft before writing in owner order. Return
        confirmed writes and any failed/uncertain prefix in a named receipt.
        Invalid/stale candidates and admission/transport errors propagate.
        This ordinary call does not ask; it never retries or rolls back.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class RecipeDeclaration:
    """Catalog declaration returned by @recipe, suitable for a RECIPES tuple.

    name is the unique recipe tool name. description states which experiments
    and handoffs the script performs. inputs is the FormModel class inferred
    from the authored function's second parameter annotation. The declaration
    retains the function for the host without exposing an untyped call API.
    No import-time registry or GUI operation is performed.
    """

    name: str
    description: str
    inputs: type[BaseModel]
    _body: Callable[[RecipeSession, BaseModel], RecipeGenerator[BaseModel]] = field(
        repr=False
    )

    def generator(
        self, session: RecipeSession, inputs: BaseModel
    ) -> RecipeGenerator[BaseModel]:
        """Bind host-validated inputs without driving the authored generator.

        The host validates inputs against inputs before constructing its session.
        The private callable retains the generic Inputs/Report pairing and
        rechecks that model at this reflection boundary; author errors propagate.
        This method is for execution hosts, not a session convenience operation.
        """
        return self._body(session, inputs)


class RecipeDecorator(Protocol):
    """Decorator that infers a function's inputs and report model separately.

    The callable receives (RecipeSession, Inputs) and returns a generator that
    returns Report or None. The result is a declaration, not a GUI execution.
    """

    def __call__[I: BaseModel, R: BaseModel](
        self, function: Callable[[RecipeSession, I], RecipeGenerator[R]]
    ) -> RecipeDeclaration:
        """Capture function and infer I from its second parameter annotation.

        function must be an annotated generator function. Missing/wrong inputs
        annotations or invalid report signatures raise TypeError. Capturing the
        declaration does not execute its body or populate a global registry.
        """
        raise NotImplementedError


def recipe(name: str, description: str) -> RecipeDecorator:
    """Declare a recipe tool and infer inputs from its decorated function.

    name and description are nonblank strings; invalid values raise ValueError.
    Use @recipe('t1', description='...') on (session, inputs: T1Inputs), where
    T1Inputs is an author-owned FormModel. Return RecipeGenerator[T1Report].
    The resulting RecipeDeclaration belongs in an explicit RECIPES tuple.
    No adapter argument, input-schema dict, summary declaration, standard
    overrides field or implicit registration is accepted. Inputs use BaseModel's
    schema/validation contract, without importing GUI Form.
    """
    if not name.strip() or not description.strip():
        raise ValueError("Recipe name and description must be nonblank")
    return _RecipeDecorator(name, description)


@dataclass(frozen=True)
class _RecipeDecorator:
    name: str
    description: str

    def __call__[I: BaseModel, R: BaseModel](
        self, function: Callable[[RecipeSession, I], RecipeGenerator[R]]
    ) -> RecipeDeclaration:
        if not isgeneratorfunction(function):
            raise TypeError("Recipe must be a generator function")
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
        reports = get_args(returned)
        if (
            get_origin(returned) is not RecipeGenerator
            or len(reports) != 1
            or not isinstance(reports[0], type)
            or not issubclass(reports[0], BaseModel)
        ):
            raise TypeError("Recipe return annotation must be RecipeGenerator[Report]")

        input_model: type[I] = hints[parameters[1].name]

        def body(
            session: RecipeSession, values: BaseModel
        ) -> RecipeGenerator[BaseModel]:
            return function(session, input_model.model_validate(values))

        return RecipeDeclaration(self.name, self.description, inputs, body)
