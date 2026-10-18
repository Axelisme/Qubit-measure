"""Qt-free form input, ordinary models and editing contracts.

Hosts own captured sources, owner revisions, submission and persistence locations.
Form paths never interpret dots in a mapping key as separators.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from logging import getLogger
from typing import TYPE_CHECKING, ClassVar, Self, final

from pydantic import BaseModel, ConfigDict, JsonValue
from pydantic_core import PydanticCustomError

from zcu_tools.device import BaseDeviceInfo
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.resources.context.library import LibrarySnapshot

if TYPE_CHECKING:
    from .commands import (
        FormCommandContext,
        FormCommandResult,
        FormCommandSnapshot,
        FormConversionResult,
        ReadonlyFormInput,
    )
    from .description import FormDescription
    from .options import Option

_logger = getLogger(__name__)


type FormPath = tuple[str | int, ...]
"""Relative model-field/mapping-key error path; integers are sequence indexes."""
type FormWritePath = tuple[str | int, ...]
"""Model-field path; only a FixedList field may be followed by an element index.

Other sequence indexes and mapping-key writes are rejected.
"""
type FormValue = (
    str
    | int
    | float
    | complex
    | bool
    | None
    | list[FormValue]
    | tuple[FormValue, ...]
    | dict[str, FormValue]
)
"""Ordinary evaluated data, including whole sequences and their nested values."""


class FormCommandError(ValueError):
    """An unavailable action or unconvertible participating input; no write occurs."""


class FormConversionError(FormCommandError):
    """Expected selection refusal; the complete previous field stays installed.

    Converters raise this for user-correctable missing sources or unsupported
    selections. Core also translates failed participating evaluation and result
    admission into this class, preserving the cause. Arbitrary callable
    ValueError and unexpected provider/programming faults are not translated.
    Editors may display this reason and restore the committed selector without
    retrying a write, converter or notification.
    """


class FormPathError(ValueError):
    """A requested model-field path is unknown or structurally unreachable.

    path is the full requested tuple of model-field names, not only the prefix
    where traversal failed. message explains an unknown field, inactive branch
    or traversal into a whole leaf. FormState.describe and run_command raise
    this type for these request errors. Other declaration, source and command
    failures keep their original exception types. Catch ValueError to include
    this category, or FormPathError to handle only correctable path requests.
    """

    def __init__(self, path: FormWritePath, message: str) -> None:
        """Store the full requested path and use message as the exception text."""
        super().__init__(message)
        self.path = path


@dataclass(frozen=True)
class InputCommand:
    """A mounted-field action replacing only that field and descendants.

    name/label identify the action. inputs names participating relative paths;
    views reject local parse errors only on those paths. run receives a readonly
    field snapshot and local evaluate context, and returns a complete named
    replacement. It must not mutate sources, state or UI. result_tag limits
    availability to annotations accepting that (field, literal value) branch.
    Declare in field metadata or on an active FormModel.input_commands tuple.
    """

    name: str
    label: str
    inputs: tuple[FormWritePath, ...]
    run: Callable[[FormCommandSnapshot, FormCommandContext], FormCommandResult]
    result_tag: tuple[str, str] | None = None


@dataclass(frozen=True)
class InputConverter:
    """One field's selection conversion, declared in metadata or converters(env).

    run receives the readonly old field, readonly requested selection and local
    evaluation context. It returns the complete new structure/input, never a
    different selection. Captured sources must be bound by the declaring host;
    call synchronously within that capture's lifetime. Exceptions reject the
    entire write; ordinary invalid values remain editable drafts.
    """

    run: Callable[
        [FormCommandSnapshot, ReadonlyFormInput, FormCommandContext],
        FormConversionResult,
    ]


@dataclass(frozen=True)
class InputPolicy:
    """Input capability restriction on an Annotated field.

    expression=False disables Expr envelopes in descriptions, evaluation,
    defaults and compatible restore. True retains the annotation's capability;
    it cannot enable expressions on categorical or container fields.
    """

    expression: bool = True


@dataclass(frozen=True)
class FixedList:
    """Annotated metadata for a list[Model] whose element identities are fixed.

    The captured input names each element with its name_field, a str field of
    the element model. The names and their order stay fixed for the life of one
    capture: writes, whole-list replacement and installed inputs may edit
    element values but cannot add, remove, rename or reorder elements. A
    duplicate name is rejected. Names must be literal strings. The core knows no domain
    meaning of a name; editors address an element by its original index.
    """

    name_field: str


@dataclass(frozen=True)
class Participants:
    """Annotated metadata for a list[str] naming FixedList elements.

    list_path is the model-field path, from the model holding this field, to
    the FixedList. Entries must name existing elements, be unique and the list
    must not be empty. Its order is independent of element order. Participation
    never filters validation: every element is validated.
    """

    list_path: tuple[str, ...]


@dataclass(frozen=True)
class Display:
    """Presentation metadata carried by Annotated fields.

    label overrides the field title (a model section's default title is its
    field name, e.g. ro_cfg -> "Ro cfg"). description explains the input; unit
    is its display unit and placeholder is an input hint. None leaves each
    item unspecified. collapsed is an initial fold hint only for model fields.
    decimals is how many digits a float field shows after the decimal point,
    rounded with trailing zeros trimmed; None shows the full value. It never
    rounds input. A negative count raises ValueError here; describe raises
    ValueError when the field is not a float. hidden=True keeps the field out
    of the editor only: its value and defaults stay in the input, it is still
    validated, and the description still lists it. group names a foldable
    display group within the field's model level, such as "Advanced"; the
    description supplies the field group name and the editor draws the
    level's groups after its ungrouped fields. None leaves the field
    ungrouped; an empty name raises ValueError here. Groups never change
    validation, defaults, choices or input.
    selector=False hides the dropdown, retaining its description and active
    branch. Explicit commands then provide the editing action.
    Declaration order sets order; nested models set sections. No layout registry.
    """

    label: str | None = None
    description: str | None = None
    unit: str | None = None
    placeholder: str | None = None
    collapsed: bool = False
    decimals: int | None = None
    hidden: bool = False
    group: str | None = None
    selector: bool = True

    def __post_init__(self) -> None:
        """Reject a negative decimals count or an empty group with ValueError."""
        if self.decimals is not None and self.decimals < 0:
            raise ValueError(f"Display decimals must be >= 0, got {self.decimals}")
        if self.group == "":
            raise ValueError("Display group must be a nonempty name or None")


@dataclass(frozen=True)
class FormEnv:
    """Captured sources borrowed read-only by defaults, choices and evaluation.

    parameters is the host's fixed ParameterContext, or None for source-free
    input and arithmetic. Parameter expressions use native point/library paths.
    library is the fixed LibrarySnapshot, or None when unused. devices maps
    host device names to captured detached BaseDeviceInfo snapshots, including
    custom driver infos, or is empty when unused. Shared tools read only the
    names; a form that needs driver-specific fields narrows with isinstance.
    The frozen shell does not make these borrowed objects deeply immutable.
    Tools never mutate, load, refresh or poll them. A used missing capability
    fails with its field path, rather than being treated as an unfilled value.
    """

    parameters: ParameterContext | None = None
    library: LibrarySnapshot | None = None
    devices: Mapping[str, BaseDeviceInfo] = field(default_factory=dict)


@dataclass(frozen=True)
class FormErrors:
    """Detached error messages by relative field path.

    messages maps model fields/mapping keys/sequence indexes to human-readable
    messages. An empty path () means a model-wide error not directed through
    field_error. Result/changes use this type only when messages is nonempty.
    """

    messages: Mapping[FormPath, str]


@dataclass(frozen=True)
class FormPreview:
    """Detached evaluated leaves and complete errors for view presentation.

    values maps active model-field leaf paths to ordinary decoded values.
    Successful validation includes Pydantic coercions and static defaults.
    On failure, only present input leaves that decoded successfully appear;
    a container with any decoding failure is omitted as a whole. Missing keys
    do not mean null. Model-invalid values may appear and are not submittable.
    errors is the complete result error set, or None for a valid model.
    Neither field aliases state or captured sources. This is not an Apply result.
    """

    values: Mapping[FormWritePath, FormValue]
    errors: FormErrors | None


@dataclass(frozen=True)
class FormChange:
    """One changed scalar, whole sequence or typed mapping relative to base.

    path is a model-field leaf path, never a sequence index or mapping key.
    input is detached JSON including expression/complex envelopes. value is
    ordinary evaluated data. Empty typed mappings remain whole leaves.
    Only current active input leaves are reported. Removed branch fields have
    no tombstone; structural changes retain their discriminator or null leaf.
    Ordinary model containers are not leaves.
    """

    path: FormWritePath
    input: JsonValue
    value: FormValue


class WriteRejected(Exception):
    """A host's write entry point refuses one input batch it will not install.

    message is the human-readable reason, written by the host, that the user
    sees at the edited field. Raise it before writing any of the batch, so the
    state is unchanged; for example for an unknown captured key. Values the
    model would reject belong in state instead, where result reports them.
    Core FormState never raises this type.
    """

    def __init__(self, message: str) -> None:
        """Store message as both the exception text and the message field."""
        super().__init__(message)
        self.message = message


def field_error(path: FormPath, message: str) -> PydanticCustomError:
    """Return an error to raise in a Pydantic model_validator.

    path is relative to that model; nested validation prefixes its parent path.
    message is the field's error text. Use ordinary ValueError for a model-wide
    error. FormState translates this error without exposing a new error owner.
    """
    return PydanticCustomError(
        "form_field", "{message}", {"path": path, "message": message}
    )


class FormModel(BaseModel):
    """Ordinary values with typed class-level input hooks.

    Override defaults for initial input (an Expr default is
    evaluated into its value), and choices for dynamic path-to-options
    mappings. default_input is final.
    No instances carry input, environment, revision or editing state. Ordinary
    nested BaseModel types remain supported without inheriting FormModel.
    default_input and FormInput reject untagged str | Model
    fields with ValueError naming the field path, even in inactive branches.
    Use a tagged ref/inline union instead.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    input_commands: ClassVar[tuple[InputCommand, ...]] = ()
    """Actions mounted on occurrences of this active model, not its parent."""

    @classmethod
    @final
    def default_input(cls, env: FormEnv) -> FormInput[Self]:
        """Build isolated input from declared defaults; required keys stay absent.

        Calls defaults(env) before checking sources and assembling input.
        Used missing library or device sources raise field_error with the
        consuming path. A text-field {"expr": text} default (int, float,
        complex or str, a choice leaf included) is evaluated once with
        env.parameters and stored as its direct value, so the input holds no
        envelope for it; later source changes need a new default_input. When
        that expression fails (missing path, unbound alias, non-numeric value)
        or its value is invalid for the field, the field is left unfilled: a
        required field then reports only "Field required" from result, and a
        field with a static Pydantic default validates with it. Other
        evaluation exceptions, such as unavailable captured sources, propagate.
        Complex JSON tags stay as they are, and so does an envelope on a
        Literal, Enum, bool or library-key leaf, which result then reports
        at that leaf. No I/O occurs.
        """
        from ._evaluate import evaluate_defaults
        from ._sources import check_sources
        from ._structure import check_form_shape

        check_form_shape(cls)
        data = cls.defaults(env)
        check_sources(cls, data, env)
        return FormInput(cls, evaluate_defaults(cls, data, env))

    @classmethod
    def defaults(cls, env: FormEnv) -> dict[str, JsonValue]:
        """Return initial input; converters may explicitly reuse this policy.

        Defaults depend only on the declaration and fixed env, never a prior
        draft. The fallback uses Pydantic static defaults and leaves required
        keys absent; it does not validate a partial model or replace missing
        keys with None. Source failures propagate at the consuming field path.
        A text-field leaf may be {"expr": text} to derive its initial value from
        env, such as {"expr": "$flux_dev"}; default_input
        evaluate it into a direct value (see default_input), so it never stays
        in input as an expression.
        """
        from .defaults import static_defaults

        del env
        return static_defaults(cls)

    @classmethod
    def check_sources(cls, data: Mapping[str, JsonValue], env: FormEnv) -> None:
        """Check this model's consumed captured sources, without mutation.

        data is this level's raw authored input, including Expr envelopes.
        env is a borrowed fixed capture. Do not recurse or validate ordinary
        values: core traverses active children, including used static defaults.
        Raise field_error(relative_path, message) for directed source failures;
        an empty path means this whole model. Unexpected provider errors propagate.
        The default consumes no sources.
        """

    @classmethod
    def choices(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        """Return dynamic legal values by relative field path; empty means none.

        Paths contain model-field names and target scalar or whole-container
        leaves, including fields inside nullable/tagged-union models. They do
        not address mapping keys, sequence indexes or ordinary model sections.
        describe and result reject invalid declared paths with ValueError.
        Validation and descriptions share these options, without restricting
        another field that reuses the same nested model. When enclosing and
        nested FormModels declare options for the same leaf, the deepest local
        declaration wins, including empty options. Nonoverlapping enclosing
        options remain in effect. A choice leaf takes one of its options,
        not an Expr envelope: describe reports expression=False and
        result reports an envelope there without evaluating it.
        Empty options for one declared path permit no value. Consumers needing
        an absent captured device/library source raise field_error at that path.
        """
        return {}

    @classmethod
    def converters(cls, _env: FormEnv, /) -> Mapping[FormPath, InputConverter]:
        """Declare converters at paths relative to this model occurrence.

        Local metadata overrides inherited paths; this model's hook overrides
        its metadata at the same path. Exactly one effective converter applies. Hooks only bind fixed
        host capabilities; they must not convert, refresh or mutate sources.
        A returned converter is used synchronously in the same host turn/env.
        """
        return {}

    @classmethod
    def options(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[Option, ...]]:
        """Declare ordered named selections by relative model-field path.

        Paths address fields, including model/union sections, not sequence
        indexes or mapping keys. Unknown paths raise ValueError in describe and
        command execution. The deepest nested declaration wins, including an
        empty tuple, which means no selections and never falls back to a
        parent or automatic list. Nonoverlapping parent paths remain effective.
        Automatic Literal/Enum, choices and tagged-union selections apply only
        without a declaration for that field. Commands resolve only the current
        active branch; inactive branch declarations cannot provide an action.
        Declarations do not add validation: Pydantic and choices remain authoritative. Read env only as a fixed
        borrowed capture, never refresh or mutate it; source failures propagate.
        The default hook is empty. Consumers read the effective
        options in FormState.describe, not this declaration hook or union structure.
        """
        return {}


class FormInput[F: FormModel]:
    """A form class paired with a detached, possibly invalid JSON input tree."""

    def __init__(
        self,
        model: type[F],
        data: Mapping[str, JsonValue],
        *,
        selections: Mapping[FormWritePath, JsonValue] | None = None,
    ) -> None:
        """Copy JSON input for model; do not require model validity.

        selections carries detached in-memory option choices independently of
        child data; omitted paths infer their selection from data. Missing keys
        remain missing and None remains explicit null. Text-field
        expressions (int, float, complex and str leaves; see
        FormState.describe) use {"expr": text}; ordinary strings remain
        literal. Complex literals use {"__complex__": [re, im]}.
        Structural selectors require ordinary typed tags, not Expr envelopes.
        Non-JSON/nonfinite data raises ValueError; unknown fields, malformed tags
        and invalid values remain input for result/compatible restoration.
        A FixedList with two elements of one name raises ValueError naming the
        original indices.
        """
        from ._fixed import check_unique_names
        from ._json import copy_object
        from ._structure import check_form_shape

        check_form_shape(model)
        self._model = model
        self._data = copy_object(data)
        check_unique_names(model, self._data)
        self._selections = {
            path: copy_object({"value": value})["value"]
            for path, value in (selections or {}).items()
        }

    @property
    def selections(self) -> dict[FormWritePath, JsonValue]:
        """Detached in-memory selections differing from the raw child branch.

        Keys are canonical field paths. Values are declared option values.
        Omitted paths infer selection from raw input. Prepared inputs and saved
        envelopes retain these choices; hosts own persistence locations.
        """
        from ._json import copy_json

        return {path: copy_json(value) for path, value in self._selections.items()}

    def same_as(self, other: FormInput[FormModel]) -> bool:
        """Compare complete authored drafts without evaluating or converting.

        Model classes compare by identity. Raw data and detached selections
        compare by content, including selection-only differences and the actual
        branch tags held in data. This is not lowered cfg or persistent Spec
        equivalence. No source access, mutation or notification occurs.
        """
        return (
            self._model is other._model
            and self._data == other._data
            and self._selections == other._selections
        )

    @property
    def model(self) -> type[F]:
        """Return the ordinary form class paired with this input."""
        return self._model

    @property
    def data(self) -> dict[str, JsonValue]:
        """Return a detached JSON tree; changing it cannot edit this input."""
        from ._json import copy_object

        return copy_object(self._data)

    def to_json(self) -> str:
        """Encode the full draft with a trusted declaration identity digest.

        Retain independent selections, raw branch tags, missing/null, expressions
        and ordinary invalid values. No evaluation, converter, callable or model
        class serialization occurs. Hosts own the destination and save timing.
        """
        from ._persistence import encode

        return encode(self._model, self.data, self.selections)

    @staticmethod
    def from_json[Model: FormModel](model: type[Model], text: str) -> FormInput[Model]:
        """Restore a full draft against the caller's trusted model declaration.

        Malformed/nonfinite envelopes and incompatible type identities raise
        ValueError. Raw invalid input remains invalid input, never an overlay on
        defaults. Branches come from model annotations and authored tags, not
        saved classes. No converter, evaluation, hook or source access occurs.
        Source availability/family remain ordinary result errors. There is no
        legacy raw-tree decoder. The result type follows model independently
        of the wrapper's type parameter, keeping FormInput covariant.
        """
        from ._persistence import decode

        data, selections = decode(model, text)
        return FormInput(model, data, selections=selections)


class FormState[F: FormModel]:
    """Single-threaded input/base editing state; hosts own submission policy."""

    def __init__(self, base: FormInput[F]) -> None:
        """Copy base into both input and comparison base without evaluation."""
        self._model = base.model
        self._data = base.data
        self._base = base.data
        self._selections = base.selections
        self._base_selections = base.selections
        self._callbacks: list[Callable[[], None]] = []

    @property
    def input(self) -> FormInput[F]:
        """Return a detached current input; no environment access occurs."""
        return FormInput(self._model, self._data, selections=self._selections)

    @property
    def base(self) -> FormInput[F]:
        """Return a detached comparison base supplied by the host."""
        return FormInput(self._model, self._base, selections=self._base_selections)

    @property
    def is_dirty(self) -> bool:
        """Whether the complete authored draft differs from the accepted base.

        Include selection-only differences and invalid/missing raw input.
        This pure observation uses FormInput.same_as, without evaluating input,
        executing converters, reading sources or publishing notifications.
        """
        return not self.input.same_as(self.base)

    def write(self, batch: Mapping[FormWritePath, JsonValue], env: FormEnv) -> None:
        """Install one input batch, keeping model-invalid drafts editable.

        Selector/tag changes apply before other leaves. A field's single
        converter receives its readonly old draft, requested selection and local
        evaluation context. Core admits the complete replacement with the same
        installer used by commands, without supplying defaults. Without a
        converter, changing a model selection preserves its raw child branch
        and input; describe reports selection and branch separately. Scalar
        selections remain ordinary leaf writes. Partial selector objects apply
        explicitly supplied child edits after conversion. Unknown structures,
        tags or child fields reject the entire batch. Writes use
        model-field leaf paths. Sequences other than FixedList and typed
        mappings are whole leaves; replace a mapping to delete, rename or clear
        its keys. Other sequence-index and mapping-key patches raise ValueError. Ordinary model replacement is
        rejected. A first child-leaf write into an absent non-nullable nested
        model creates its static defaults object before applying the leaf.
        Nullable model sections accept None or an object to deactivate/activate;
        an explicit null section requires object activation before child edits.
        Nonempty
        partial union objects write supplied children; ordinary non-nullable
        model replacement outside a union remains rejected.
        Structurally unknown or overlapping paths raise ValueError without
        installing a partial batch.
        Ordinary invalid values/Expr are retained for result to report. Missing
        used library/devices raises field_error naming its consuming path.
        FixedList elements are edited as ("list", index, field) paths. A write
        that changes their names, count or order, even by whole-list
        replacement, raises ValueError and installs nothing.
        Notify subscribers once after a changed batch is installed, not for
        intermediate branches. No Engine update, revision check or I/O occurs.
        """
        self.install_input(self.prepare_write(batch, env))

    def prepare_write(
        self, batch: Mapping[FormWritePath, JsonValue], env: FormEnv
    ) -> FormInput[F]:
        """Build an isolated candidate using write's structural/source admission.

        Ordinary model-invalid values remain editable. Structural ValueError or
        directed source errors leave state untouched. No subscribers run and
        base is preserved. Use only in the same owner turn/current input/env,
        without crossing await, reset or guard boundaries; not an authorization
        or revision token. install_input does not repeat defaults/source checks.
        """
        from ._transitions import prepare_write

        return prepare_write(self.input, batch, env)

    def install_input(self, candidate: FormInput[F]) -> None:
        """Install a same-model prepared input, preserving base.

        A different model raises TypeError, and a candidate that changes the
        FixedList element names, count or order raises ValueError. Changed content notifies subscribers
        once after installation; equal content is a no-op. Hosts guard the whole
        prepare/install turn. This does not repeat validation or source checks.
        """
        if candidate.model is not self._model:
            raise TypeError("Cannot install a different form model class")
        self._install(candidate)

    def describe(self, path: FormWritePath, env: FormEnv) -> FormDescription:
        """Describe one canonical model-field path and its next layer.

        path is a tuple of field names and FixedList element indexes, () for root. env is the host's fixed
        capture for choices/options hooks. Return detached named metadata, not
        JSON Schema or evaluated values. Children have full paths and no
        descendants. Each field's can_expand reports structural reachability
        without calling child hooks, not a promise of nonempty children. A
        selected option's lone ordinary model wrapper flattens one display
        layer without introducing a path alias. Ordinary missing
        models retain declared fields. Disabled nullable models and unmatched
        options expose metadata only, never guessed defaults. Unknown paths,
        inactive descendants, invalid or out-of-range FixedList indexes and
        traversal into whole leaves raise FormPathError with the full requested path. Unsupported declarations and other
        non-path failures keep their original exception types.
        Does not validate, evaluate expressions, call defaults, read sources
        outside declaration hooks, mutate input/base or notify subscribers.
        """
        from ._describe import describe

        return describe(self._model, self._data, path, env, self._selections)

    def run_command(self, path: FormWritePath, name: str, env: FormEnv) -> None:
        """Execute a declared field action or a selected option's named action.

        path addresses an active field; name comes from its commands or the
        selected option's commands. Both replace only the mounted field, with
        no fresh defaults. Callables receive a readonly snapshot and a context
        that evaluates only requested participants. FormCommandError rejects
        unavailable actions, failed participants and illegal structure atomically.
        Resolve declarations with the same fixed env used by the host for writes
        and sources. Unknown paths, inactive
        descendants and whole-leaf traversal raise FormPathError with the full
        requested path. Missing current input, no matching option or unavailable
        commands raise ValueError, never silently do nothing. Other failures
        keep their original exception types. The callable returns a complete
        replacement for path; install it atomically and notify once.
        Callable/source/structural failures leave state unchanged; ordinary
        validation failures remain drafts for result/preview. No refresh,
        library semantics, I/O policy or separate state is introduced.
        """
        from ._commands import prepare_command

        self.install_input(prepare_command(self.input, path, name, env))

    def _install(self, candidate: FormInput[F]) -> None:
        from ._fixed import require_same_identity

        require_same_identity(self._model, self._data, candidate.data)
        if self.input.same_as(candidate):
            return
        self._data = candidate.data
        self._selections = candidate.selections
        for callback in tuple(self._callbacks):
            try:
                callback()
            except Exception:
                _logger.exception("Form state subscriber failed")

    def result(self, env: FormEnv) -> F | FormErrors:
        """Evaluate/decode input and validate its model, without editing input.

        Reuse the scalar evaluator with env.parameters and the existing complex codec.
        Only text fields evaluate {"expr": text}. An envelope on a choice leaf
        is an error at that leaf; on a Literal, Enum, bool or library-key
        leaf it is ordinary input that Pydantic rejects there. Neither is
        evaluated. Validate dynamic choices, explicit selection overrides
        against current options, and Pydantic types/ranges/cross-field rules.
        Raw branches without an override do not acquire an options constraint.
        Return a model or path-directed errors, including a used missing source;
        ordinary model-wide errors use (). Each call uses the supplied env.
        Input at a field whose union mixes text and picked members raises
        ValueError naming its path, as describe does.
        """
        from ._evaluate import result

        return result(self._model, self._data, env, self._selections)

    def preview(self, env: FormEnv) -> FormPreview:
        """Evaluate once for display, preserving partial values on model errors.

        Return detached active leaves and the same errors as result(env).
        Failed expression/container decoding has no value, not a stale value
        or a null substitute. This does not edit input, run domain defaults,
        submit, poll sources, or establish validity beyond preview.errors.
        Structural/choice declaration failures propagate as for result.
        """
        from ._evaluate import preview

        return preview(self._model, self._data, env, self._selections)

    def changes(self, env: FormEnv) -> tuple[FormChange, ...] | FormErrors:
        """Return changed input leaves and their evaluated values, or errors.

        Compare input with base, preserving expression-only changes even when
        the evaluated value is unchanged. Sequences and typed mappings are
        whole leaves, including empty mappings. Report only leaves present in
        the current active input, including discriminator changes and nullable
        null; removed branch fields have no tombstones. Ordinary models are not
        leaves. Invalid input returns FormErrors, never partial changes.
        This is a data/evaluated-leaf report, not a dirty predicate: selection-only
        differences are not data leaves. Use is_dirty for complete authored
        identity. This does not reset base or commit to an owner. Uses env on every call.
        """
        from ._evaluate import value_at
        from ._json import copy_json
        from ._structure import leaves

        result = self.result(env)
        if isinstance(result, FormErrors):
            return result
        before = leaves(self._model, self._base)
        changes: list[FormChange] = []
        for path, value in leaves(self._model, self._data).items():
            if path in before and value == before[path]:
                continue
            evaluated = value_at(result, path)
            changes.append(FormChange(path, copy_json(value), evaluated))
        return tuple(changes)

    def reset(self, base: FormInput[F]) -> None:
        """Replace input/base with detached base and notify once when changed.

        A different form class raises ValueError. Discard/accepted Apply choose
        when to call this; reset does not evaluate, submit or change revision.
        """
        if base.model is not self._model:
            raise ValueError("Cannot reset to a different form model class")
        data = base.data
        changed = not self.input.same_as(base) or not self.base.same_as(base)
        self._data = data
        self._base = base.data
        self._selections = base.selections
        self._base_selections = base.selections
        if changed:
            for callback in tuple(self._callbacks):
                callback()

    def subscribe(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Register a plain post-write/reset callback and return unsubscribe.

        Unsubscribe is idempotent. Callbacks run synchronously and can observe
        installed input. Individual callback failures log a traceback and do not
        roll back input or prevent later subscribers or host publication.
        No Qt signals, lifecycle controller or owner adapter is created.
        """
        self._callbacks.append(callback)
        active = True

        def unsubscribe() -> None:
            nonlocal active
            if active:
                active = False
                self._callbacks.remove(callback)

        return unsubscribe

    def write_compatible(
        self, saved: Mapping[str, JsonValue], env: FormEnv
    ) -> tuple[FormPath, ...]:
        """Overlay surviving compatible saved leaves and return dropped paths.

        Start with host-provided current default input. Keep valid Expr/complex
        tags, missing-vs-null and whole sequence/typed-mapping leaves. Drop removed paths
        and type-incompatible leaves, but keep range/cross-field-invalid values
        so result can report them. Untagged model unions select a branch only
        when saved field names and raw value types match exactly one member;
        valid expression envelopes count as compatible for text fields without
        evaluation; on a Literal, Enum, bool or library-key leaf an envelope
        is incompatible and dropped. Choice leaves are not checked here. Zero or multiple matches discard that subtree,
        never select the first branch. Tagged selection/defaults stay unchanged.
        This is not legacy format support. Branch
        writes use the same discriminator-first order as write. A saved object
        without a discriminator overlays the current selected branch, retaining
        its tag and unsaved defaults; an empty object makes no change. If neither
        saved nor current input selects a branch, drop that object instead of
        replacing the model section. A restored leaf whose captured source check
        fails, such as a ref name absent from env.library or any ref without a
        captured library, keeps the current input at that leaf (or stays unfilled
        when current has none) and its leaf path is returned with the other
        dropped paths. Source failures that no restored leaf causes, such as a
        used static default or a model-wide error, raise PydanticCustomError
        without installing. A changed restoration notifies once.
        """
        from ._json import copy_object
        from ._sources import drop_unmatched_sources
        from ._structure import restore

        restored, dropped = restore(self._model, self._data, copy_object(saved), env)
        candidate, unmatched = drop_unmatched_sources(
            self._model, restored, self._data, env
        )
        self._install(FormInput(self._model, candidate))
        return (*dropped, *unmatched)
