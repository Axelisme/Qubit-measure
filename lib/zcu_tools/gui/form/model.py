"""Qt-free form input, ordinary models and editing contracts.

Hosts own captured sources, owner revisions, submission and persistence locations.
Form paths never interpret dots in a mapping key as separators.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue
from pydantic_core import PydanticCustomError

from zcu_tools.cfg_model import ModuleSpec
from zcu_tools.device import DeviceInfo
from zcu_tools.resources.context import MetaDict
from zcu_tools.resources.context.library import LibrarySnapshot

type FormPath = tuple[str | int, ...]
"""Relative model-field/mapping-key error path; integers are sequence indexes."""
type FormWritePath = tuple[str, ...]
"""Model-field/mapping-key path; sequence index writes are not supported."""
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


@dataclass(frozen=True)
class Display:
    """Presentation metadata carried by Annotated fields.

    label overrides the field-name label. description explains the input; unit
    is its display unit and placeholder is an input hint. None leaves each
    item unspecified. collapsed is an initial fold hint only for model fields.
    Declaration order sets order; nested models set groups. No layout registry.
    """

    label: str | None = None
    description: str | None = None
    unit: str | None = None
    placeholder: str | None = None
    collapsed: bool = False


@dataclass(frozen=True)
class FormEnv:
    """Captured sources borrowed read-only by defaults, choices and evaluation.

    md is a host-provided MetaDict memory snapshot, not a live file store.
    library is the fixed LibrarySnapshot, or None when unused. devices maps
    host device names to captured native DeviceInfo, or is empty when unused.
    The frozen shell does not make these borrowed objects deeply immutable.
    Tools never mutate, load, refresh or poll them. A used missing capability
    fails with its field path, rather than being treated as an unfilled value.
    """

    md: MetaDict
    library: LibrarySnapshot | None = None
    devices: Mapping[str, DeviceInfo] = field(default_factory=dict)


@dataclass(frozen=True)
class FormErrors:
    """Detached error messages by relative field path.

    messages maps model fields/mapping keys/sequence indexes to human-readable
    messages. An empty path () means a model-wide error not directed through
    field_error. Result/changes use this type only when messages is nonempty.
    """

    messages: Mapping[FormPath, str]


@dataclass(frozen=True)
class FormChange:
    """One changed scalar or whole sequence relative to the input base.

    path is a model-field/mapping-key path, never a sequence index. input is
    detached JSON including its expression/complex envelope. value is ordinary
    evaluated data. original is the scalar expression text without its envelope,
    or None for direct input; it is suitable for the Engine's opaque original.
    """

    path: FormWritePath
    input: JsonValue
    value: FormValue
    original: str | None


def field_error(path: FormPath, message: str) -> PydanticCustomError:
    """Return an error to raise in a Pydantic model_validator.

    path is relative to that model; nested validation prefixes its parent path.
    message is the field's error text. Use ordinary ValueError for a model-wide
    error. FormState translates this error without exposing a new error owner.
    """
    raise NotImplementedError("F1 internal implementation pending")


class FormModel(BaseModel):
    """Ordinary values with typed class-level input hooks.

    Override default_input for domain defaults (including Expr), choices for
    dynamic path-to-options mappings, and defaults for a branch-switch factory.
    No instances carry input, environment, revision or editing state. Ordinary
    nested BaseModel types remain supported without inheriting FormModel.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    @classmethod
    def default_input(cls, env: FormEnv) -> FormInput[Self]:
        """Build isolated input from declared defaults; required keys stay absent.

        Domain overrides may include Expr/complex JSON. Used missing library or
        device sources raise field_error with the consuming path. No I/O occurs.
        """
        raise NotImplementedError("F1 internal implementation pending")

    @classmethod
    def defaults(
        cls, env: FormEnv, previous: Mapping[str, JsonValue]
    ) -> dict[str, JsonValue]:
        """Return new-branch defaults before same-name inputs are inherited.

        previous is detached old-branch JSON. The fallback uses Pydantic static
        defaults and leaves required keys absent; it does not validate a partial
        model or replace missing keys with None. Domain factories may use env
        and previous; source failures propagate with the consuming field path.
        """
        raise NotImplementedError("F1 internal implementation pending")

    @classmethod
    def choices(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        """Return dynamic legal values by relative field path; empty means none.

        These options are shared by validation and the input schema. Empty
        options for one declared path permit no value. Consumers requiring an
        absent captured device/library source raise field_error at that path.
        """
        return {}

    @classmethod
    def input_schema(cls, env: FormEnv) -> dict[str, JsonValue]:
        """Describe accepted input without constructing editing state.

        Include numeric Expr alternatives, the shared complex tag, Field
        constraints, Display metadata and current choices enums. This schema
        describes partial JSON input, not just a valid ordinary model. Domain
        source failures from choices propagate; no refresh or file access occurs.
        """
        raise NotImplementedError("F1 internal implementation pending")


class FormInput[F: FormModel]:
    """A form class paired with a detached, possibly invalid JSON input tree."""

    def __init__(self, model: type[F], data: Mapping[str, JsonValue]) -> None:
        """Copy JSON input for model; do not require model validity.

        Missing keys remain missing and None remains explicit null. Numeric
        expressions use {"expr": text}; complex uses {"__complex__": [re, im]}.
        Non-JSON/nonfinite data raises ValueError; unknown fields, malformed tags
        and invalid values remain input for result/compatible restoration.
        """
        raise NotImplementedError("F1 internal implementation pending")

    @property
    def model(self) -> type[F]:
        """Return the ordinary form class paired with this input."""
        raise NotImplementedError("F1 internal implementation pending")

    @property
    def data(self) -> dict[str, JsonValue]:
        """Return a detached JSON tree; changing it cannot edit this input."""
        raise NotImplementedError("F1 internal implementation pending")

    def to_json(self) -> str:
        """Encode only input, losslessly retaining missing/null and scalar tags."""
        raise NotImplementedError("F1 internal implementation pending")

    @staticmethod
    def from_json[Model: FormModel](model: type[Model], text: str) -> FormInput[Model]:
        """Decode saved JSON for model; malformed/non-object/nonfinite JSON raises.

        This static factory's result type follows model, independently of the
        wrapper's type parameter, keeping immutable FormInput covariant.
        Model compatibility is deliberately deferred to write_compatible so old
        field names or changed leaf types can be reported instead of hidden.
        """
        raise NotImplementedError("F1 internal implementation pending")


class FormState[F: FormModel]:
    """Single-threaded input/base editing state; hosts own submission policy."""

    def __init__(self, base: FormInput[F]) -> None:
        """Copy base into both input and comparison base without evaluation."""
        raise NotImplementedError("F1 internal implementation pending")

    @property
    def input(self) -> FormInput[F]:
        """Return a detached current input; no environment access occurs."""
        raise NotImplementedError("F1 internal implementation pending")

    @property
    def base(self) -> FormInput[F]:
        """Return a detached comparison base supplied by the host."""
        raise NotImplementedError("F1 internal implementation pending")

    def write(self, batch: Mapping[FormWritePath, JsonValue], env: FormEnv) -> None:
        """Install one input batch, keeping model-invalid drafts editable.

        Discriminator changes apply first: use the new branch's defaults(env,
        previous), then inherit old same-name fields except the discriminator.
        Apply all remaining leaf writes afterwards, so explicitly supplied
        content wins. Required fields without defaults stay absent. Writes use
        model-field/mapping-key paths; dots within a key have no special meaning.
        Whole sequences are leaves. Nullable model sections accept None or an
        object to deactivate/activate the section before child leaf edits.
        Index patches and structurally unknown or overlapping paths raise
        ValueError without installing a partial batch.
        Ordinary invalid values/Expr are retained for result to report. Missing
        used library/devices raises field_error naming its consuming path.
        Notify subscribers once after a changed batch is installed, not for
        intermediate branches. No Engine update, revision check or I/O occurs.
        """
        raise NotImplementedError("F1 internal implementation pending")

    def result(self, env: FormEnv) -> F | FormErrors:
        """Evaluate/decode input and validate its model, without editing input.

        Reuse the scalar evaluator with env.md and the existing complex codec.
        Validate dynamic choices and Pydantic types/ranges/cross-field rules.
        Return a model or path-directed errors, including a used missing source;
        ordinary model-wide errors use (). Each call uses the supplied env.
        """
        raise NotImplementedError("F1 internal implementation pending")

    def changes(self, env: FormEnv) -> tuple[FormChange, ...] | FormErrors:
        """Return changed input leaves and their evaluated values, or errors.

        Compare input with base, preserving expression-only changes even when
        the evaluated value is unchanged. Lists/tuples are whole leaves. Invalid
        input returns FormErrors rather than a partially valid change batch.
        This does not reset base or commit to an owner. Uses env on every call.
        """
        raise NotImplementedError("F1 internal implementation pending")

    def reset(self, base: FormInput[F]) -> None:
        """Replace input/base with detached base and notify once when changed.

        A different form class raises ValueError. Discard/accepted Apply choose
        when to call this; reset does not evaluate, submit or change revision.
        """
        raise NotImplementedError("F1 internal implementation pending")

    def subscribe(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Register a plain post-write/reset callback and return unsubscribe.

        Unsubscribe is idempotent. Callbacks can observe installed input;
        callback errors propagate after installation and do not roll it back.
        No Qt signals, lifecycle controller or owner adapter is created.
        """
        raise NotImplementedError("F1 internal implementation pending")

    def input_schema(self, env: FormEnv) -> dict[str, JsonValue]:
        """Delegate the form class's input_schema; state is not schema authority."""
        raise NotImplementedError("F1 internal implementation pending")

    def write_compatible(
        self, saved: Mapping[str, JsonValue], env: FormEnv
    ) -> tuple[FormPath, ...]:
        """Overlay surviving compatible saved leaves and return dropped paths.

        Start with host-provided current default input. Keep valid Expr/complex
        tags, missing-vs-null and complete sequence leaves. Drop removed paths
        and type-incompatible leaves, but keep range/cross-field-invalid values
        so result can report them. This is not legacy format support. Branch
        writes use the same discriminator-first order as write. Source failures
        retain their consuming path; a changed restoration notifies once.
        """
        raise NotImplementedError("F1 internal implementation pending")


class LinRange(FormModel):
    """Stored linear range; adapters expand it only when converting to cfg.

    kind selects this branch. start/stop are finite endpoints in caller units;
    n is the positive number of points (one point is permitted). Descending or
    repeated endpoints are allowed; no step/expts synchronization occurs.
    """

    kind: Literal["linrange"] = "linrange"
    start: float = 0.0
    stop: float = 1.0
    n: int = Field(default=3, ge=1)


class ExplicitList(FormModel):
    """Stored explicit finite values in caller units and their declared order.

    kind selects this branch. values is a nonempty whole sequence; adapters
    decide how to use it. Inputs persist as JSON arrays, models hold tuples.
    """

    kind: Literal["list"] = "list"
    values: tuple[float, ...] = Field(default=(0.0,), min_length=1)


type RangeInput = Annotated[LinRange | ExplicitList, Field(discriminator="kind")]
"""Stored range or explicit-list branches; no implicit array expansion."""


class ModuleRef(FormModel):
    """Library module reference; no inline override or cached materialization.

    kind selects ref. name is an actual LibrarySnapshot module key, not a role
    or alias. A used ref needs captured library; missing/invalid names fail at
    the consuming path through public library reads, never fall back to defaults.
    """

    kind: Literal["ref"] = "ref"
    name: str


class ModuleInline(FormModel):
    """Detached inline Spec, not a runtime cfg or linked library cache.

    kind selects inline. module is the ordinary cfg_model ModuleSpec preserving
    its own references. Adapter to_cfg owns materialization/family requirements.
    """

    kind: Literal["inline"] = "inline"
    module: ModuleSpec

    @classmethod
    def defaults(
        cls, env: FormEnv, previous: Mapping[str, JsonValue]
    ) -> dict[str, JsonValue]:
        """Seed inline from previous ref's captured library Spec when switching.

        previous uses {kind: ref, name: actual_key}. Read LibrarySnapshot's public
        module_spec and retain a detached input under module. Missing captured
        library/key or invalid module family fails at the consuming field path;
        it never supplies a dummy Spec. Static defaults apply without a prior ref.
        """
        raise NotImplementedError("F1 internal implementation pending")


type ModuleInput = Annotated[ModuleRef | ModuleInline, Field(discriminator="kind")]
"""Reference or inline Spec with wire {kind: inline, module: {...}}."""
