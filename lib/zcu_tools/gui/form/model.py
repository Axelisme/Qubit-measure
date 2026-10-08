"""Qt-free form input, ordinary models and editing contracts.

Hosts own captured sources, owner revisions, submission and persistence locations.
Form paths never interpret dots in a mapping key as separators.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue
from pydantic_core import PydanticCustomError

from zcu_tools.cfg_model import ModuleSpec
from zcu_tools.device import DeviceInfo
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.resources.context.library import LibrarySnapshot

type FormPath = tuple[str | int, ...]
"""Relative model-field/mapping-key error path; integers are sequence indexes."""
type FormWritePath = tuple[str, ...]
"""Model-field leaf path; sequence indexes and mapping-key writes are rejected."""
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

    parameters is the host's fixed ParameterContext, or None for source-free
    input and arithmetic. Parameter expressions use native point/library paths.
    library is the fixed LibrarySnapshot, or None when unused. devices maps
    host device names to captured native DeviceInfo, or is empty when unused.
    The frozen shell does not make these borrowed objects deeply immutable.
    Tools never mutate, load, refresh or poll them. A used missing capability
    fails with its field path, rather than being treated as an unfilled value.
    """

    parameters: ParameterContext | None = None
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
    ordinary evaluated data. original is scalar expression text without its
    envelope, or None for direct input and whole containers; it is suitable
    for the Engine's opaque original. Empty typed mappings remain whole leaves.
    Only current active input leaves are reported. Removed branch fields have
    no tombstone; structural changes retain their discriminator or null leaf.
    Ordinary model containers are not leaves.
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
    return PydanticCustomError(
        "form_field", "{message}", {"path": path, "message": message}
    )


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
        from ._sources import check_sources

        data = cls.defaults(env, {})
        check_sources(cls, data, env)
        return FormInput(cls, data)

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
        from ._structure import static_defaults

        del env, previous
        return static_defaults(cls)

    @classmethod
    def choices(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        """Return dynamic legal values by relative field path; empty means none.

        Paths contain model-field names and target scalar or whole-container
        leaves, including fields inside nullable/tagged-union models. They do
        not address mapping keys, sequence indexes or ordinary model sections.
        input_schema and result reject invalid declared paths with ValueError.
        Validation and input schema share these options, without restricting
        another field that reuses the same nested model. When enclosing and
        nested FormModels declare options for the same leaf, the deepest local
        declaration wins, including empty options. Nonoverlapping enclosing
        options remain in effect. Numeric/complex leaves permit a well-formed
        arithmetic Expr; result validates its evaluated
        value against choices. The reused evaluator does not return strings.
        Empty options for one declared path permit no value. Consumers needing
        an absent captured device/library source raise field_error at that path.
        """
        return {}

    @classmethod
    def input_schema(cls, env: FormEnv) -> dict[str, JsonValue]:
        """Describe accepted input without constructing editing state.

        Include numeric Expr alternatives, the shared complex tag, Field
        constraints, Display metadata and current choices enums. Each tagged
        oneOf branch has x-form-discriminator-values: its intrinsic Literal
        tags as ordinary JSON values, independent of current choices. This
        schema describes partial JSON input, not just a valid model. Domain
        source failures from choices propagate; no refresh or file access occurs.
        """
        from ._schema import input_schema

        return input_schema(cls, env)


class FormInput[F: FormModel]:
    """A form class paired with a detached, possibly invalid JSON input tree."""

    def __init__(self, model: type[F], data: Mapping[str, JsonValue]) -> None:
        """Copy JSON input for model; do not require model validity.

        Missing keys remain missing and None remains explicit null. Numeric
        expressions use {"expr": text}; complex uses {"__complex__": [re, im]}.
        Non-JSON/nonfinite data raises ValueError; unknown fields, malformed tags
        and invalid values remain input for result/compatible restoration.
        """
        from ._json import copy_object

        self._model = model
        self._data = copy_object(data)

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
        """Encode only input, losslessly retaining missing/null and scalar tags."""
        return json.dumps(self._data, allow_nan=False)

    @staticmethod
    def from_json[Model: FormModel](model: type[Model], text: str) -> FormInput[Model]:
        """Decode saved JSON for model; malformed/non-object/nonfinite JSON raises.

        This static factory's result type follows model, independently of the
        wrapper's type parameter, keeping immutable FormInput covariant.
        Model compatibility is deliberately deferred to write_compatible so old
        field names or changed leaf types can be reported instead of hidden.
        """
        from ._json import copy_object

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON input: {exc}") from exc
        return FormInput(model, copy_object(data))


class FormState[F: FormModel]:
    """Single-threaded input/base editing state; hosts own submission policy."""

    def __init__(self, base: FormInput[F]) -> None:
        """Copy base into both input and comparison base without evaluation."""
        self._model = base.model
        self._data = base.data
        self._base = base.data
        self._callbacks: list[Callable[[], None]] = []

    @property
    def input(self) -> FormInput[F]:
        """Return a detached current input; no environment access occurs."""
        return FormInput(self._model, self._data)

    @property
    def base(self) -> FormInput[F]:
        """Return a detached comparison base supplied by the host."""
        return FormInput(self._model, self._base)

    def write(self, batch: Mapping[FormWritePath, JsonValue], env: FormEnv) -> None:
        """Install one input batch, keeping model-invalid drafts editable.

        Discriminator changes apply first: use the new branch's defaults(env,
        previous), then inherit old same-name fields except the discriminator.
        Apply all remaining leaf writes afterwards, so explicitly supplied
        content wins. Required fields without defaults stay absent. Writes use
        model-field leaf paths. Sequences and typed mappings are whole leaves;
        replace a mapping to delete, rename or clear its keys. Mapping-key and
        sequence-index patches raise ValueError. Ordinary model replacement is
        rejected. A first child-leaf write into an absent non-nullable nested
        model creates its static defaults object before applying the leaf.
        Nullable model sections accept None or an object to deactivate/activate;
        an explicit null section requires object activation before child edits.
        Structurally unknown or overlapping paths raise ValueError without
        installing a partial batch.
        Ordinary invalid values/Expr are retained for result to report. Missing
        used library/devices raises field_error naming its consuming path.
        Notify subscribers once after a changed batch is installed, not for
        intermediate branches. No Engine update, revision check or I/O occurs.
        """
        from ._sources import check_sources
        from ._structure import build_candidate

        candidate = build_candidate(self._model, self._data, batch, env)
        check_sources(self._model, candidate, env)
        self._install(candidate)

    def _install(self, candidate: dict[str, JsonValue]) -> None:
        if candidate == self._data:
            return
        self._data = candidate
        for callback in tuple(self._callbacks):
            callback()

    def result(self, env: FormEnv) -> F | FormErrors:
        """Evaluate/decode input and validate its model, without editing input.

        Reuse the scalar evaluator with env.parameters and the existing complex codec.
        Validate dynamic choices and Pydantic types/ranges/cross-field rules.
        Return a model or path-directed errors, including a used missing source;
        ordinary model-wide errors use (). Each call uses the supplied env.
        """
        from ._evaluate import result

        return result(self._model, self._data, env)

    def preview(self, env: FormEnv) -> FormPreview:
        """Evaluate once for display, preserving partial values on model errors.

        Return detached active leaves and the same errors as result(env).
        Failed expression/container decoding has no value, not a stale value
        or a null substitute. This does not edit input, run domain defaults,
        submit, poll sources, or establish validity beyond preview.errors.
        Structural/choice declaration failures propagate as for result.
        """
        from ._evaluate import preview

        return preview(self._model, self._data, env)

    def changes(self, env: FormEnv) -> tuple[FormChange, ...] | FormErrors:
        """Return changed input leaves and their evaluated values, or errors.

        Compare input with base, preserving expression-only changes even when
        the evaluated value is unchanged. Sequences and typed mappings are
        whole leaves, including empty mappings. Report only leaves present in
        the current active input, including discriminator changes and nullable
        null; removed branch fields have no tombstones. Ordinary models are not
        leaves. Invalid input returns FormErrors, never partial changes.
        This does not reset base or commit to an owner. Uses env on every call.
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
            original = (
                value.get("expr")
                if isinstance(value, dict)
                and not isinstance(evaluated, (dict, list, tuple))
                else None
            )
            changes.append(
                FormChange(
                    path,
                    copy_json(value),
                    evaluated,
                    original if isinstance(original, str) else None,
                )
            )
        return tuple(changes)

    def reset(self, base: FormInput[F]) -> None:
        """Replace input/base with detached base and notify once when changed.

        A different form class raises ValueError. Discard/accepted Apply choose
        when to call this; reset does not evaluate, submit or change revision.
        """
        if base.model is not self._model:
            raise ValueError("Cannot reset to a different form model class")
        data = base.data
        changed = data != self._data or data != self._base
        self._data = data
        self._base = base.data
        if changed:
            for callback in tuple(self._callbacks):
                callback()

    def subscribe(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Register a plain post-write/reset callback and return unsubscribe.

        Unsubscribe is idempotent. Callbacks can observe installed input;
        callback errors propagate after installation and do not roll it back.
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

    def input_schema(self, env: FormEnv) -> dict[str, JsonValue]:
        """Delegate the form class's input_schema; state is not schema authority."""
        return self._model.input_schema(env)

    def write_compatible(
        self, saved: Mapping[str, JsonValue], env: FormEnv
    ) -> tuple[FormPath, ...]:
        """Overlay surviving compatible saved leaves and return dropped paths.

        Start with host-provided current default input. Keep valid Expr/complex
        tags, missing-vs-null and whole sequence/typed-mapping leaves. Drop removed paths
        and type-incompatible leaves, but keep range/cross-field-invalid values
        so result can report them. This is not legacy format support. Branch
        writes use the same discriminator-first order as write. A saved object
        without a discriminator overlays the current selected branch, retaining
        its tag and unsaved defaults; an empty object makes no change. If neither
        saved nor current input selects a branch, drop that object instead of
        replacing the model section. Source failures retain their consuming
        path; a changed restoration notifies once.
        """
        from ._json import copy_object
        from ._sources import check_sources
        from ._structure import restore

        candidate, dropped = restore(self._model, self._data, copy_object(saved), env)
        check_sources(self._model, candidate, env)
        self._install(candidate)
        return dropped


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
        from ._json import copy_object
        from ._structure import static_defaults

        data = static_defaults(cls)
        if previous.get("kind") == "ref" and isinstance(
            name := previous.get("name"), str
        ):
            if env.library is None:
                raise field_error(
                    ("module",), "Missing captured library for inline module"
                )
            try:
                spec = env.library.module_spec(name)
            except (ValueError, KeyError) as exc:
                raise field_error(
                    ("module",), f"Cannot read module {name!r}: {exc}"
                ) from exc
            data["module"] = copy_object(spec.model_dump(mode="json"))
        return data


type ModuleInput = Annotated[ModuleRef | ModuleInline, Field(discriminator="kind")]
"""Reference or inline Spec with wire {kind: inline, module: {...}}."""
