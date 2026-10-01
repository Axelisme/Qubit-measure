from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import InitVar, dataclass, field, replace
from typing import Any, Self, TypeAlias


def default_value_for_type(type_: type) -> object:
    defaults: dict[type, object] = {int: 0, float: 0.0, bool: False, str: ""}
    return defaults.get(type_)


def require_finite_scalar(value: float | complex) -> None:
    if not (math.isfinite(value.real) and math.isfinite(value.imag)):
        raise ValueError("Scalar value must be finite")


# ---------------------------------------------------------------------------
# Spec tree — static, defined by Adapter, never mutated
# ---------------------------------------------------------------------------


# A transform applied to the leaf spec node reached by a dotted path. Returns a
# replacement node (e.g. a LiteralSpec for lock_literal).
def _split_spec_path(path: str) -> list[str]:
    parts = [p for p in path.split(".") if p]
    if not parts:
        raise RuntimeError("Spec override path must not be empty")
    return parts


def _path_exists(spec: CfgSectionSpec, parts: list[str]) -> bool:
    """True if the dotted ``parts`` resolve to a leaf within ``spec`` (descending
    CfgSectionSpec.fields and ReferenceSpec.allowed). Used by the
    duck-type descent to decide which allowed shapes contain a path."""
    head, rest = parts[0], parts[1:]
    child = spec.fields.get(head)
    if child is None:
        return False
    if not rest:
        return True
    if isinstance(child, CfgSectionSpec):
        return _path_exists(child, rest)
    if isinstance(child, ReferenceSpec):
        return any(_path_exists(shape, rest) for shape in child.allowed)
    return False


@dataclass(frozen=True)
class ScalarSpec:
    label: str
    type: type
    editable: bool = True
    choices: Sequence[object] | None = None
    choices_source: str = ""
    decimals: int | None = None
    required: bool = False
    # ``optional``: the field may be left empty (value ``None``) and is *valid*
    # while empty — at lowering an unset optional scalar is omitted so the model
    # default (typically ``None``) applies (e.g. PulseCfg.mixer_freq). This is
    # the opposite of ``required`` (which forces a value: empty = invalid), so
    # the two are mutually exclusive.
    optional: bool = False
    # ``group``: pure presentation hint — fields sharing a non-empty group label
    # render together under a collapsible sub-header (e.g. "Advanced"). It does
    # NOT nest the value tree; the field stays a flat leaf of its section.
    group: str = ""
    tooltip: str = ""

    def __post_init__(self) -> None:
        if self.required and self.optional:
            raise RuntimeError(
                f"ScalarSpec {self.label!r}: 'required' and 'optional' are "
                "mutually exclusive"
            )


def IntSpec(
    label: str,
    *,
    editable: bool = True,
    choices: Sequence[object] | None = None,
    required: bool = False,
    optional: bool = False,
    group: str = "",
    tooltip: str = "",
) -> ScalarSpec:
    """Sugar for ``ScalarSpec(label=..., type=int)`` — an integer field.

    A thin, explicit factory (not a default ``type``): callers see ``IntSpec``
    and know it is int, with no hidden default to remember. Mirrors the
    ``ScalarSpec`` fields relevant to integers (``decimals`` is float-only).
    """
    return ScalarSpec(
        label=label,
        type=int,
        editable=editable,
        choices=choices,
        required=required,
        optional=optional,
        group=group,
        tooltip=tooltip,
    )


def FloatSpec(
    label: str,
    *,
    decimals: int | None = None,
    editable: bool = True,
    choices: Sequence[object] | None = None,
    required: bool = False,
    optional: bool = False,
    group: str = "",
    tooltip: str = "",
) -> ScalarSpec:
    """Sugar for ``ScalarSpec(label=..., type=float)`` — a float field.

    Explicit counterpart to :func:`IntSpec`; carries the float-only ``decimals``.
    """
    return ScalarSpec(
        label=label,
        type=float,
        decimals=decimals,
        editable=editable,
        choices=choices,
        required=required,
        optional=optional,
        group=group,
        tooltip=tooltip,
    )


@dataclass(frozen=True)
class LiteralSpec:
    """A fixed-value field: no widget shown, value is always spec.value."""

    value: Any
    label: str = ""


@dataclass(frozen=True)
class SweepSpec:
    label: str = "Sweep"
    editable: bool = True
    decimals: int | None = None
    tooltip: str = ""


@dataclass(frozen=True)
class CenteredSweepSpec:
    label: str = "Sweep"
    editable: bool = True
    decimals: int | None = None
    tooltip: str = ""
    center_editable: bool = True
    center_badge: str = ""
    center_tooltip: str = ""
    locked_center: float | None = None

    def __post_init__(self) -> None:
        if self.locked_center is not None and not math.isfinite(
            float(self.locked_center)
        ):
            raise RuntimeError("CenteredSweepSpec.locked_center must be finite")


@dataclass(frozen=True)
class ReferenceSpec:
    kind: str
    allowed: list[CfgSectionSpec]
    label: str = "Reference"
    optional: bool = False
    discriminator: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if not self.kind:
            raise RuntimeError("ReferenceSpec.kind must be non-empty")
        if not self.allowed:
            raise RuntimeError("ReferenceSpec.allowed must be non-empty")
        if self.discriminator is not None:
            values: list[object] = []
            for shape in self.allowed:
                literal = shape.fields.get(self.discriminator)
                if not isinstance(literal, LiteralSpec):
                    raise ValueError(
                        "Reference discriminator must name a literal in every shape"
                    )
                if literal.value in values:
                    raise ValueError(
                        "Reference discriminator must uniquely identify each shape"
                    )
                values.append(literal.value)

    def lock_literal(self, path: str, value: object) -> Self:
        """Lock a leaf of this ref's allowed shapes (path is relative to the
        shape, e.g. ``pulse_cfg.freq``). Lets an adapter lock fields on the
        sub-tree as it is built, instead of from the root section. Returns a new
        frozen ReferenceSpec; chains stay on this type."""
        parts = _split_spec_path(path)
        new_allowed: list[CfgSectionSpec] = []
        matched = False
        for shape in self.allowed:
            if _path_exists(shape, parts):
                new_allowed.append(shape.lock_literal(path, value))
                matched = True
            else:
                new_allowed.append(shape)
        if not matched:
            allowed_labels = ", ".join(s.label for s in self.allowed)
            raise RuntimeError(
                f"Spec override path {'.'.join(parts)!r} not found in any allowed "
                f"shape of ReferenceSpec (allowed: {allowed_labels})"
            )
        return replace(self, allowed=new_allowed)


@dataclass(frozen=True)
class CfgSectionSpec:
    fields: dict[str, CfgNodeSpec] = field(default_factory=dict)
    label: str = ""
    inherit_hook: (
        Callable[[CfgSectionValue, CfgSectionSpec], CfgSectionValue | None] | None
    ) = None

    # -- fluent spec overrides (return a new frozen spec; never mutate) -------
    #
    # Used while authoring a context-free cfg definition to lock/restrict a deep
    # leaf. Locking is part of the spec contract and stays local to the field
    # declaration that owns it.

    def lock_literal(self, path: str, value: object) -> Self:
        """Replace the scalar leaf at ``path`` with a fixed ``LiteralSpec(value)``.

        The locked field shows no widget and always lowers to ``value`` (notebook
        ``freq: 0.0, # not used``). Returns a new frozen spec.
        """
        head, *rest = _split_spec_path(path)
        if head not in self.fields:
            raise RuntimeError(
                f"Spec override path segment {head!r} not found "
                f"(available: {', '.join(self.fields)})"
            )
        child = self.fields[head]
        if not rest:
            new_child: CfgNodeSpec = LiteralSpec(value=value)
        elif isinstance(child, (CfgSectionSpec, ReferenceSpec)):
            new_child = child.lock_literal(".".join(rest), value)
        else:
            raise RuntimeError(
                f"Spec override path cannot descend into {type(child).__name__} "
                f"at segment {head!r}"
            )
        return replace(self, fields={**self.fields, head: new_child})


@dataclass(frozen=True)
class ChoiceBinding:
    """One selector-driven variant list inside a ChoiceSectionSpec.

    ``choices`` maps a selector value to the section spec whose fields should be
    visible for that value. The owner ChoiceSectionSpec still owns the complete
    union ``fields``; the choice specs are the display contract.
    """

    selector_key: str
    choices: Mapping[str, CfgSectionSpec]

    def __post_init__(self) -> None:
        if not self.selector_key:
            raise RuntimeError("ChoiceBinding.selector_key must be non-empty")
        if not self.choices:
            raise RuntimeError("ChoiceBinding.choices must be non-empty")

    def controlled_field_keys(self) -> set[str]:
        keys: set[str] = set()
        for spec in self.choices.values():
            keys.update(spec.fields)
        return keys


@dataclass(frozen=True)
class ChoiceSectionSpec(CfgSectionSpec):
    """A section that renders selector-specific child specs.

    The value tree remains a normal complete CfgSectionValue over the union of
    ``fields``. Only rendering is variant-aware: selector fields stay editable, and
    fields listed by inactive choice specs are omitted from the form.
    """

    bindings: tuple[ChoiceBinding, ...] = ()

    def __post_init__(self) -> None:
        if not self.bindings:
            raise RuntimeError("ChoiceSectionSpec.bindings must be non-empty")
        field_keys = set(self.fields)
        for binding in self.bindings:
            if binding.selector_key not in field_keys:
                raise RuntimeError(
                    f"Choice selector {binding.selector_key!r} is not a section field"
                )
            for choice, spec in binding.choices.items():
                unknown = set(spec.fields) - field_keys
                if unknown:
                    raise RuntimeError(
                        f"Choice {binding.selector_key!r}={choice!r} references "
                        "unknown field(s): " + ", ".join(sorted(unknown))
                    )
                if binding.selector_key in spec.fields:
                    raise RuntimeError(
                        f"Choice {binding.selector_key!r}={choice!r} must not include "
                        "its own selector field"
                    )


CfgNodeSpec = (
    ScalarSpec
    | LiteralSpec
    | SweepSpec
    | CenteredSweepSpec
    | ReferenceSpec
    | CfgSectionSpec
    | ChoiceSectionSpec
)


# ---------------------------------------------------------------------------
# Value tree — mutable, holds user-editable state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DirectValue:
    """A directly-entered scalar value with optional input text and parse error.

    ``value=None`` means unset when ``error`` is absent. Scalar types are
    int/float/complex/str/bool, whose legal values are never ``None``.
    Text parsing failures retain ``raw``
    and ``error`` with no parsed value, so invalid input cannot reuse an earlier
    valid value. ``validation_error`` records binding constraints such as option
    membership without discarding the current selection. Binding recomputes that
    runtime metadata; persistence stores only the input.
    The wrapper also preserves direct mode while unset or invalid."""

    value: Any | None = None
    raw: str | None = field(default=None, kw_only=True)
    error: str | None = field(default=None, kw_only=True)
    validation_error: str | None = field(default=None, kw_only=True, repr=False)

    def __post_init__(self) -> None:
        if self.error is not None and (self.raw is None or self.value is not None):
            raise ValueError(
                "Invalid direct input requires raw text and no parsed value"
            )


@dataclass(frozen=True)
class EvalValue:
    expr: str
    resolved: Any | None = None
    error: str | None = None
    validation_error: str | None = field(default=None, kw_only=True, repr=False)


ScalarValue: TypeAlias = DirectValue | EvalValue

# Accepted input for the value-tree fluent ``with_field``: a raw scalar (wrapped
# in DirectValue) or an already-built scalar value.
ScalarLeafInput: TypeAlias = (
    int | float | complex | str | bool | DirectValue | EvalValue
)


def resolved_direct_number(value: int | float | DirectValue) -> int | float | None:
    """Read a numeric carrier without reparsing text or substituting an old value."""
    number = value.value if isinstance(value, DirectValue) else value
    if number is None:
        return None
    if isinstance(number, bool) or not isinstance(number, (int, float)):
        raise TypeError("Expected a direct numeric value")
    return number


def _normalize_range_text(
    value: DirectValue, label: str, type_: type, minimum: float | None = None
) -> DirectValue:
    number = resolved_direct_number(value)
    if number is None:
        return value
    if type(number) is not type_:
        raise TypeError(f"{label} expects {type_.__name__}")
    try:
        finite = math.isfinite(number)
    except OverflowError:
        finite = False
    if finite and (minimum is None or number >= minimum):
        return value
    error = f"{label} must be finite"
    if minimum is not None:
        error += f" and >= {minimum:g}"
    if value.raw is None:
        raise ValueError(error)
    return DirectValue(None, raw=value.raw, error=error)


def _normalize_sweep_points(value: int | DirectValue) -> int | DirectValue:
    if isinstance(value, DirectValue):
        return _normalize_range_text(value, "Sweep points", int, 1)
    if type(value) is not int:
        raise TypeError("Sweep points must be an integer")
    if value < 1:
        raise ValueError("SweepValue.expts must be >= 1")
    return value


@dataclass
class SweepValue:
    start: float | ScalarValue
    stop: float | ScalarValue
    expts: int | DirectValue
    step: float | DirectValue = 0.1
    # ``auto_norm`` (init-only) derives ``step`` from start/stop/expts at
    # construction so that any direct ``SweepValue(start, stop, expts=N)`` (the
    # 16 adapter defaults, session codec, inheritance) is self-consistent — step
    # is a derived view of expts, not an independent input. ``SweepEditor`` (the
    # canonicalisation authority, which also runs the reverse step→expts rule)
    # passes ``auto_norm=False`` so its already-computed value is not re-derived.
    # Numeric direct bounds are normalised; EvalValue bounds are left to
    # ``SweepEditor`` (which owns the resolved-edge handling) — auto_norm never
    # touches an EvalValue's ``resolved`` (it may be unresolved or non-numeric).
    auto_norm: InitVar[bool] = True

    def __post_init__(self, auto_norm: bool) -> None:
        self.expts = _normalize_sweep_points(self.expts)
        if isinstance(self.start, DirectValue):
            self.start = _normalize_range_text(self.start, "Sweep start", float)
        if isinstance(self.stop, DirectValue):
            self.stop = _normalize_range_text(self.stop, "Sweep stop", float)
        if isinstance(self.step, DirectValue):
            self.step = _normalize_range_text(self.step, "Sweep step", float)
        start = self.start.value if isinstance(self.start, DirectValue) else self.start
        stop = self.stop.value if isinstance(self.stop, DirectValue) else self.stop
        points = resolved_direct_number(self.expts)
        if (
            auto_norm
            and not isinstance(self.step, DirectValue)
            and points is not None
            and isinstance(start, (int, float))
            and isinstance(stop, (int, float))
        ):
            self.step = (
                0.0 if points == 1 else (float(stop) - float(start)) / (points - 1)
            )


@dataclass
class CenteredSweepValue:
    center: float | ScalarValue
    span: float | DirectValue
    expts: int | DirectValue
    step: float | DirectValue = 0.1
    auto_norm: InitVar[bool] = True

    def __post_init__(self, auto_norm: bool) -> None:
        self.expts = _normalize_sweep_points(self.expts)
        if isinstance(self.center, DirectValue):
            self.center = _normalize_range_text(self.center, "Sweep center", float)
        if isinstance(self.span, DirectValue):
            self.span = _normalize_range_text(self.span, "Sweep span", float, 0)
        else:
            span = float(self.span)
            if not math.isfinite(span) or span < 0.0:
                raise ValueError("CenteredSweepValue.span must be finite and >= 0")
            self.span = span
        if isinstance(self.step, DirectValue):
            self.step = _normalize_range_text(self.step, "Sweep step", float, 0)
        span = resolved_direct_number(self.span)
        points = resolved_direct_number(self.expts)
        if (
            auto_norm
            and not isinstance(self.step, DirectValue)
            and span is not None
            and points is not None
        ):
            self.step = 0.0 if points == 1 else span / (points - 1)


@dataclass
class ReferenceValue:
    chosen_key: str
    value: CfgSectionValue
    # True when chosen_key names a library entry but the user has edited value
    # away from the library snapshot (LibraryBindingState.MODIFIED). Persisted so
    # the override survives reload; False for pure library refs and <Custom:> refs.
    is_overridden: bool = False
    # Resolution metadata belongs to the displayed snapshot, not the live catalog.
    resolved_label: str | None = field(default=None, kw_only=True)
    error: str | None = field(default=None, kw_only=True)

    def with_field(self, path: str, value: ScalarLeafInput) -> Self:
        """Set a scalar leaf inside this ref's value (in-place, returns self).

        Adapter-side default override sugar (replaces long factory params). The
        value tree is mutable by contract; this mutates and returns self for
        chaining — deliberately asymmetric with spec-side fluent (which returns
        new frozen specs). See the gui/cfg README for Spec/Value ownership.
        """
        self.value.with_field(path, value)
        return self


@dataclass
class CfgSectionValue:
    # The value tree is always *complete*: every spec field has a corresponding
    # entry (no missing keys, ADR-0010). A disabled optional ModuleRef/WaveformRef
    # is represented by ``None`` (the entry is present, its value is None) — never
    # by omitting the key. "None" here means "this optional ref is not enabled",
    # distinct from a "None Reset" library entry (a real, enabled reset choice).
    fields: dict[str, CfgNodeValue | None] = field(default_factory=dict)

    def with_field(self, path: str, value: ScalarLeafInput) -> Self:
        """Set the scalar leaf at dotted ``path`` (in-place, returns self).

        ``value`` may be a raw scalar (wrapped in ``DirectValue``) or an already-
        built ``DirectValue``/``EvalValue``. Descends ``CfgSectionValue.fields``
        and ``ReferenceValue`` (into its ``.value``).
        """
        from .tree import replace_value_path

        leaf_value: CfgNodeValue = (
            value if isinstance(value, (DirectValue, EvalValue)) else DirectValue(value)
        )
        replace_value_path(self, path, leaf_value)
        return self


CfgNodeValue = (
    ScalarValue | SweepValue | CenteredSweepValue | ReferenceValue | CfgSectionValue
)


# ---------------------------------------------------------------------------
# CfgSchema — pairs a spec tree with a value tree
# ---------------------------------------------------------------------------


@dataclass
class CfgSchema:
    spec: CfgSectionSpec
    value: CfgSectionValue
