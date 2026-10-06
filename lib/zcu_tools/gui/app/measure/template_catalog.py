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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeAlias, TypedDict

from zcu_tools.experiment.cfg_editing import PROGRAM_SHAPES, UnknownProgramShapeError
from zcu_tools.gui.cfg import CfgSectionSpec, LiteralSpec, ReferenceValue

from .adapter import SessionEnv

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
    the live ``SessionEnv`` and returns a fresh ``ReferenceValue`` carrying
    md-linked ``EvalValue`` defaults. Create calls both factories fresh.

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
        _validate_entry_shape(entry)
        self._entries[entry.template_id] = entry

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


def _validate_entry_shape(entry: TemplateEntry) -> None:
    spec = entry.shape()
    if not isinstance(spec, CfgSectionSpec):
        raise TypeError(
            f"Template {entry.template_id!r} shape factory must return CfgSectionSpec, "
            f"got {type(spec).__name__}"
        )
    key = "type" if entry.item_kind == "module" else "style"
    other_kind: TemplateItemKind = (
        "waveform" if entry.item_kind == "module" else "module"
    )
    other_key = "style" if entry.item_kind == "module" else "type"
    if key in spec.fields and other_key in spec.fields:
        raise ValueError(
            f"Template {entry.template_id!r} shape must declare exactly one root discriminator"
        )
    literal = spec.fields.get(key)
    if not isinstance(literal, LiteralSpec) or not isinstance(literal.value, str):
        other_literal = spec.fields.get(other_key)
        if isinstance(other_literal, LiteralSpec) and isinstance(
            other_literal.value, str
        ):
            raise TypeError(
                f"Template {entry.template_id!r} declares kind {entry.item_kind!r} but "
                f"shape root kind is {other_kind!r}"
            )
        raise ValueError(
            f"Template {entry.template_id!r} shape has no string literal discriminator {key!r}"
        )
    discriminator = literal.value
    try:
        PROGRAM_SHAPES.get(entry.item_kind, discriminator)
    except UnknownProgramShapeError:
        try:
            PROGRAM_SHAPES.get(other_kind, discriminator)
        except UnknownProgramShapeError as exc:
            raise ValueError(
                f"Template {entry.template_id!r} has unknown {entry.item_kind} shape "
                f"{discriminator!r}"
            ) from exc
        raise TypeError(
            f"Template {entry.template_id!r} declares kind {entry.item_kind!r} but "
            f"shape root kind is {other_kind!r}"
        ) from None
