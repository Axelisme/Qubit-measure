"""Catalog of named templates for creating ModuleLibrary entries.

A ``TemplateEntry`` pairs a human-facing template (e.g. "Resonator probe readout")
with its eval-aware value factory from the user-owned defaults.
The GUI defines this interface; injected user composition populates it at startup
(mirroring ``Registry`` / ``register_all``), keeping the dependency direction
correct (gui must not import the adapters package).

Picking a template and a name seeds a blank ml module/waveform whose defaults are
"the thing to define" (md-linked values) rather than structural zero-values. It
is a one-shot create: editing the entry afterwards goes through the normal
modify path (inspect modify / ``editor.new(from_name=...)``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias, TypedDict

from zcu_tools.cfg_model import Expr, Slot, TemplateDef, TemplateSeed
from zcu_tools.experiment.cfg_editing import PROGRAM_SHAPES, UnknownProgramShapeError
from zcu_tools.gui.cfg import (
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    LiteralSpec,
    ReferenceSpec,
    ReferenceValue,
    make_custom_reference_key,
    make_default_value,
    select_ref_value_spec,
)

from .adapter import SessionEnv
from .reference_defaults import resolve_library_reference
from .specs import MAIN_PROGRAM_SPEC_POLICY

logger = logging.getLogger(__name__)

TemplateItemKind: TypeAlias = Literal["module", "waveform"]
TemplateShapeFactory: TypeAlias = Callable[[], CfgSectionSpec]
TemplateValueFactory: TypeAlias = Callable[[SessionEnv], ReferenceValue]


class TemplateMetadata(TypedDict):
    """Discovery metadata with the existing RPC field names.

    ``role_id`` is the template's unique ID, preserved under the wire name.
    ``label`` is its human-facing dropdown text. ``item_kind`` selects the
    ModuleLibrary store, either ``module`` or ``waveform``. ``default_name`` is
    the suggested new entry name, or an empty string for no suggestion.
    """

    role_id: str
    label: str
    item_kind: TemplateItemKind
    default_name: str


@dataclass(frozen=True, slots=True)
class TemplateEntry:
    """One immutable template for creating a ModuleLibrary entry.

    ``template_id`` is the unique catalog lookup ID, including ``:blank`` IDs.
    ``label`` is human-facing dropdown text. ``item_kind`` selects the
    ``module`` or ``waveform`` store.

    ``shape`` returns a fresh context-free canonical Spec. Registration validates
    its root discriminator without calling ``make_value``. ``make_value`` takes
    a ``SessionEnv`` whose parameters are the create operation's fixed
    alias-aware capture, and returns a fresh ``ReferenceValue`` carrying
    parameter-linked ``EvalValue`` defaults. Create calls both factories fresh
    and lowers their result with that same capture.

    ``default_name`` suggests a new entry name, e.g. ``"readout_rf"``. An empty
    string means no suggestion, so blank templates leave naming to the user.
    """

    template_id: str
    label: str
    item_kind: TemplateItemKind
    shape: TemplateShapeFactory
    make_value: TemplateValueFactory
    default_name: str = ""


class TemplateCatalog:
    """Ordered registry of ``TemplateEntry`` (insertion order = dropdown order)."""

    def __init__(self) -> None:
        self._entries: dict[str, TemplateEntry] = {}

    def register(self, entry: TemplateEntry) -> None:
        """Append an entry after validating its shape, without creating a value.

        Raise ValueError for a duplicate ID or invalid discriminator, TypeError
        for an invalid shape type/kind, and propagate shape-factory failures.
        Validation failures leave the catalog unchanged.
        """
        logger.debug(
            "register template: id=%r kind=%r", entry.template_id, entry.item_kind
        )
        if entry.template_id in self._entries:
            raise ValueError(f"Template {entry.template_id!r} is already registered")
        _validate_entry_shape(entry.template_id, entry.item_kind, entry.shape())
        self._entries[entry.template_id] = entry

    def register_definitions(self, templates: Mapping[str, TemplateDef]) -> None:
        """Append injected persistent definitions in their mapping order.

        ModuleSpec and WaveformSpec members select the matching canonical
        program shape. Each create builds fresh structural defaults, then
        applies seeds; empty seeds therefore preserve library blank behavior.
        Expr seeds become EvalValue, while ordinary strings stay literal.
        Nested reference discriminator seeds select an allowed inline shape.
        Slot(path) seeds on module references use SMART linked defaults from the
        create operation's capture: unbound aliases stay linked empty, while
        bound unset/dangling slots use the nested structural defaults. Waveform
        Slots and nondefault copy/set/lock/fallback Form options raise TypeError
        at registration; Slot with inline child seeds raises ValueError.
        Invalid seed paths/shapes raise ValueError or TypeError; duplicate IDs
        raise ValueError. Call only at startup, not on catalog reload.
        """
        for template_id, definition in templates.items():
            self.register(_entry_from_definition(template_id, definition))

    def entries_for(self, item_kind: TemplateItemKind) -> list[TemplateEntry]:
        """Return module or waveform entries in registration order, without factories."""
        return [e for e in self._entries.values() if e.item_kind == item_kind]

    def get(self, template_id: str) -> TemplateEntry:
        """Return the registered entry by ID; raise KeyError for an unknown ID."""
        if template_id not in self._entries:
            raise KeyError(
                f"Template {template_id!r} not found; available: {list(self._entries)}"
            )
        return self._entries[template_id]

    def has(self, template_id: str) -> bool:
        """Report whether the ID is registered, without invoking either factory."""
        return template_id in self._entries

    def list_meta(self) -> list[TemplateMetadata]:
        """Return fresh wire metadata in registration order, without factories."""
        return [
            {
                "role_id": e.template_id,
                "label": e.label,
                "item_kind": e.item_kind,
                "default_name": e.default_name,
            }
            for e in self._entries.values()
        ]


def _validate_entry_shape(
    template_id: str, item_kind: TemplateItemKind, spec: object
) -> None:
    if not isinstance(spec, CfgSectionSpec):
        raise TypeError(
            f"Template {template_id!r} shape factory must return CfgSectionSpec, "
            f"got {type(spec).__name__}"
        )
    key = "type" if item_kind == "module" else "style"
    other_kind: TemplateItemKind = "waveform" if item_kind == "module" else "module"
    other_key = "style" if item_kind == "module" else "type"
    if key in spec.fields and other_key in spec.fields:
        raise ValueError(
            f"Template {template_id!r} shape must declare exactly one root discriminator"
        )
    literal = spec.fields.get(key)
    if not isinstance(literal, LiteralSpec) or not isinstance(literal.value, str):
        other_literal = spec.fields.get(other_key)
        if isinstance(other_literal, LiteralSpec) and isinstance(
            other_literal.value, str
        ):
            raise TypeError(
                f"Template {template_id!r} declares kind {item_kind!r} but "
                f"shape root kind is {other_kind!r}"
            )
        raise ValueError(
            f"Template {template_id!r} shape has no string literal discriminator {key!r}"
        )
    discriminator = literal.value
    try:
        PROGRAM_SHAPES.get(item_kind, discriminator)
    except UnknownProgramShapeError:
        try:
            PROGRAM_SHAPES.get(other_kind, discriminator)
        except UnknownProgramShapeError as exc:
            raise ValueError(
                f"Template {template_id!r} has unknown {item_kind} shape "
                f"{discriminator!r}"
            ) from exc
        raise TypeError(
            f"Template {template_id!r} declares kind {item_kind!r} but "
            f"shape root kind is {other_kind!r}"
        ) from None


def _entry_from_definition(template_id: str, definition: TemplateDef) -> TemplateEntry:
    # Persistent union members have exactly one of these root discriminators.
    kind: TemplateItemKind = (
        "module" if "type" in definition.module.model_fields else "waveform"
    )
    key = "type" if kind == "module" else "style"
    discriminator = definition.module.model_fields[key].default
    if not isinstance(discriminator, str):
        raise TypeError(f"Template {template_id!r} requires a string discriminator")
    program_shape = PROGRAM_SHAPES.get(kind, discriminator)
    seeds = definition.expanded_seeds()

    def shape() -> CfgSectionSpec:
        return program_shape.make_spec(MAIN_PROGRAM_SPEC_POLICY)

    def make_value(ctx: SessionEnv) -> ReferenceValue:
        spec = shape()
        value = _seeded_section(spec, seeds, ctx)
        return ReferenceValue(make_custom_reference_key(spec.label), value)

    # Fail on malformed seeds at injection, without reading any session source.
    _seeded_section(shape(), seeds)
    return TemplateEntry(
        template_id, definition.label, kind, shape, make_value, definition.default_name
    )


def _seeded_section(
    spec: CfgSectionSpec,
    seeds: Mapping[str, TemplateSeed],
    env: SessionEnv | None = None,
) -> CfgSectionValue:
    value = make_default_value(spec)
    unknown = {path.split(".")[0] for path in seeds} - spec.fields.keys()
    if unknown:
        raise ValueError(f"Unknown template seed fields: {sorted(unknown)}")
    for key, node_spec in spec.fields.items():
        child_seeds = {
            path[len(key) + 1 :]: seed
            for path, seed in seeds.items()
            if path.startswith(f"{key}.")
        }
        seed = seeds.get(key)
        if isinstance(seed, Slot):
            if child_seeds:
                raise ValueError(f"Slot seed {key!r} cannot also have inline children")
            value.fields[key] = _slot_reference(node_spec, value.fields[key], seed, env)
            continue
        if isinstance(node_spec, ReferenceSpec) and child_seeds:
            current = value.fields[key]
            selected = _seed_reference_shape(node_spec, current, child_seeds)
            value.fields[key] = ReferenceValue(
                make_custom_reference_key(selected.label),
                _seeded_section(selected, child_seeds, env),
            )
        elif isinstance(node_spec, CfgSectionSpec) and child_seeds:
            value.fields[key] = _seeded_section(node_spec, child_seeds, env)
        elif child_seeds:
            raise TypeError(f"Template seed field {key!r} has no children")
        if key in seeds:
            if isinstance(node_spec, (ReferenceSpec, CfgSectionSpec)):
                raise TypeError(f"Template seed field {key!r} requires child paths")
            if isinstance(node_spec, LiteralSpec) and seed != node_spec.value:
                raise ValueError(f"Template seed cannot change literal {key!r}")
            value.with_field(
                key,
                EvalValue(seed.text) if isinstance(seed, Expr) else DirectValue(seed),
            )
    return value


def _slot_reference(
    spec: object, current: object, slot: Slot, env: SessionEnv | None
) -> ReferenceValue | None:
    if not isinstance(spec, ReferenceSpec):
        raise TypeError("Slot seed requires a reference field")
    if spec.kind != "module":
        raise TypeError("Library Slot seeds support only module references")
    if slot.copy or slot.set or slot.lock or slot.fallback is not None:
        raise TypeError("Library seeds support only Slot(path), without Form options")
    if not isinstance(current, ReferenceValue) and current is not None:
        raise TypeError("Slot field requires a structural reference default")
    if env is None:
        # Startup validates syntax/shape without consulting a session source.
        return current
    return resolve_library_reference(
        slot.path,
        env,
        kind="module",
        fallback=(lambda _env: current)
        if isinstance(current, ReferenceValue)
        else None,
    )


def _seed_reference_shape(
    spec: ReferenceSpec,
    current: object,
    seeds: Mapping[str, TemplateSeed],
) -> CfgSectionSpec:
    discriminator_key = spec.discriminator
    if discriminator_key is None:
        raise ValueError("Template reference shape requires a discriminator")
    discriminator = seeds.get(discriminator_key)
    if discriminator is not None:
        for allowed in spec.allowed:
            literal = allowed.fields.get(discriminator_key)
            if isinstance(literal, LiteralSpec) and literal.value == discriminator:
                return allowed
        raise ValueError(f"Unknown template seed shape {discriminator!r}")
    if isinstance(current, ReferenceValue):
        return select_ref_value_spec(spec, current)
    return spec.allowed[0]
