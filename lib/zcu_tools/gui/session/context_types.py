"""Typed parameter destinations and detached source diagnostics."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict

from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError

if TYPE_CHECKING:
    from zcu_tools.gui.cfg import CfgSchema


class SavedPoint(BaseModel):
    """A persisted destination within an entry.

    kind is the JSON discriminator, always saved_point.
    entry_id is the UUID string supplied by ResultEntry.entry_id.
    point is the entry owner's validated single-segment point label.
    This value carries no selection generation or writable handle. Commands
    validate entry membership and the label through the owning ResultEntry.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    kind: Literal["saved_point"] = "saved_point"
    entry_id: str
    point: str


class EntryDraft(BaseModel):
    """The current memory-only draft belonging to an opened entry.

    kind is the JSON discriminator, always entry_draft.
    entry_id is the UUID string supplied by ResultEntry.entry_id.
    Commands require that this entry is currently in DRAFT; the value does not
    represent a saved point or carry a binding generation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    kind: Literal["entry_draft"] = "entry_draft"
    entry_id: str


ContextTarget: TypeAlias = SavedPoint | EntryDraft
ContextSourceKind: TypeAlias = Literal["setup", "point", "module_cfg"]


@dataclass(frozen=True)
class LibraryChange:
    """One structural library command in concrete component paths.

    kind is module or waveform. action is put (create/overwrite), replace
    (overwrite an existing entry, optionally relocating it), rename (relocate
    without lowering), or delete (remove without cascading references).
    path is the destination for put and the existing source for other actions.
    new_path is required only for replace/rename; replace permits the same path.
    schema is required only for put/replace and is lowered with the caller's
    fixed ParameterContext. Invalid combinations raise InvalidInputError.
    Library owners validate path families, source existence and destinations.
    """

    kind: Literal["module", "waveform"]
    action: Literal["put", "replace", "rename", "delete"]
    path: str
    new_path: str | None = None
    schema: CfgSchema | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("module", "waveform"):
            raise InvalidInputError(f"Unknown library kind: {self.kind!r}")
        if self.action not in ("put", "replace", "rename", "delete"):
            raise InvalidInputError(f"Unknown library action: {self.action!r}")
        if (self.new_path is not None) != (self.action in ("replace", "rename")):
            raise InvalidInputError("Only replace/rename require new_path")
        if (self.schema is not None) != (self.action in ("put", "replace")):
            raise InvalidInputError("Only put/replace require schema")


@dataclass(frozen=True)
class EntryRequest:
    """Entry name: one ResultEntry-validated path segment, not a path or UUID."""

    name: str


@dataclass(frozen=True)
class EntryInfo:
    """Opened entry UUID, directory name, and absolute owner-supplied roots.

    entry_id is ResultEntry.entry_id. name is its single-segment directory name.
    result_path and database_path are the exact paths owned by this entry.
    """

    entry_id: str
    name: str
    result_path: Path
    database_path: Path


@dataclass(frozen=True)
class WorkingPointInfo:
    """A point label and its general.description, with absent description as ''."""

    label: str
    description: str


@dataclass(frozen=True)
class PointUseRequest:
    """Select label in the currently opened entry identified by its UUID."""

    entry_id: str
    label: str


@dataclass(frozen=True)
class CommandOrigin:
    """Frontend attribution, exactly gui or mcp.

    origin identifies the initiating frontend. Unknown values raise ValueError
    before any resource command.
    """

    origin: Literal["gui", "mcp"]

    def __post_init__(self) -> None:
        if self.origin not in ("gui", "mcp"):
            raise ValueError("Command origin must be gui or mcp")


@dataclass(frozen=True)
class PointCreateRequest:
    """Create a point in the opened entry, independently of its memory DRAFT.

    entry_id is the opened entry UUID. label is a safe single path segment.
    description replaces the seed/clone description, including empty text.
    origin attributes the resource command. clone_from is a same-entry label;
    None copies setup components and uses an empty library.
    """

    entry_id: str
    label: str
    description: str
    origin: CommandOrigin
    clone_from: str | None = None


class PointBindingError(RuntimeError):
    """A point was created but could not be installed; never delete or retry it.

    entry_id and label identify the retained destination. The raising service
    chains the installation cause; Open/Use is recovery, not repeating New.
    """

    entry_id: str
    label: str

    def __init__(self, *, entry_id: str, label: str) -> None:
        self.entry_id = entry_id
        self.label = label
        super().__init__(
            f"Point {label!r} in entry {entry_id!r} was created but could not be "
            "installed. Use the same point to recover; do not repeat New."
        )


@dataclass(frozen=True)
class ComponentOption:
    """Point component name and its registered kind, in document order."""

    name: str
    kind: str


@dataclass(frozen=True)
class ComponentFocus:
    """Cached session focus, detached from consumer roles and selection IDs.

    target is None for EMPTY, EntryDraft for DRAFT, or SavedPoint for ACTIVE.
    name is the selected component or None for no selection.
    components lists all current component names/kinds in document order.
    """

    target: ContextTarget | None
    name: str | None
    components: tuple[ComponentOption, ...]


@dataclass(frozen=True)
class ContextSource:
    """A bound persistent source and its last published health diagnostic.

    document is setup, point, or module_cfg. path is its absolute owner path.
    fault is None after successful opening/refresh, not a guarantee about disk.
    """

    document: ContextSourceKind
    path: Path
    fault: ContextSourceFault | None


@dataclass(frozen=True)
class ContextSourceBinding:
    """Internal session source identity and ordered persistent sources.

    binding_id is the positive session-local ID allocated on installation.
    sources is ordered setup, point, module_cfg, omitting unbound documents.
    This ID is for source facts/watcher routing, not a read/write target.
    """

    binding_id: int
    sources: tuple[ContextSource, ...]


@dataclass(frozen=True)
class ContextSourceFault:
    """Diagnostic for a source: document kind, absolute path, exception name and message."""

    document: ContextSourceKind
    path: Path
    error_type: str
    message: str


class ContextSourceUnavailableError(FailedPreconditionError):
    """Required sources are unreadable; faults are detached diagnostics.

    Accept a nonempty tuple, otherwise raise ValueError. No exception object or
    traceback is retained. The source owner may supply the original cause.
    """

    faults: tuple[ContextSourceFault, ...]

    def __init__(self, faults: tuple[ContextSourceFault, ...]) -> None:
        if not faults:
            raise ValueError("Source unavailability requires at least one fault")
        self.faults = deepcopy(faults)
        super().__init__(
            "; ".join(f"{fault.document}: {fault.message}" for fault in faults),
            reason_code="source_unavailable",
        )
