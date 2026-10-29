"""Typed parameter destinations and detached source diagnostics."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict

from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError

if TYPE_CHECKING:
    from zcu_tools.cfg_model import ModuleSpec, WaveformSpec
    from zcu_tools.resources.context.library import LibrarySnapshot


class SavedPoint(BaseModel):
    """A persisted destination within an entry.

    kind is the JSON discriminator, always saved_point.
    entry_name is ResultEntry.result_path.name within the currently opened result root.
    point is the entry owner's validated single-segment point label.
    This value carries no selection generation or writable handle. Commands
    validate entry membership and the label through the owning ResultEntry.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    kind: Literal["saved_point"] = "saved_point"
    entry_name: str
    point: str


ContextTarget: TypeAlias = SavedPoint
ContextSourceKind: TypeAlias = Literal["setup", "point", "module_cfg"]


@dataclass(frozen=True)
class LibraryChange:
    """One structural library command in concrete component paths.

    kind is module or waveform. action is put (create/overwrite), replace
    (overwrite an existing entry, optionally relocating it), rename (relocate
    an entry with its sources), or delete (remove without cascading references).
    path is the destination for put and the existing source for other actions.
    new_path is required only for replace/rename; replace permits the same path.
    module is the persistent Spec written by a module put/replace; waveform is
    the Spec written by a waveform put/replace. Exactly the field matching kind
    is set for put/replace, and neither is set for rename/delete. Child library
    keys inside the Spec stay keys. Invalid combinations raise InvalidInputError.
    Library owners validate the Spec family, path families, source existence
    and destinations.
    """

    kind: Literal["module", "waveform"]
    action: Literal["put", "replace", "rename", "delete"]
    path: str
    new_path: str | None = None
    module: ModuleSpec | None = None
    waveform: WaveformSpec | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("module", "waveform"):
            raise InvalidInputError(f"Unknown library kind: {self.kind!r}")
        if self.action not in ("put", "replace", "rename", "delete"):
            raise InvalidInputError(f"Unknown library action: {self.action!r}")
        if (self.new_path is not None) != (self.action in ("replace", "rename")):
            raise InvalidInputError("Only replace/rename require new_path")
        writes = self.action in ("put", "replace")
        if (self.module is not None) != (writes and self.kind == "module"):
            raise InvalidInputError("Only module put/replace require a module Spec")
        if (self.waveform is not None) != (writes and self.kind == "waveform"):
            raise InvalidInputError("Only waveform put/replace require a waveform Spec")

    def apply(self, library: LibrarySnapshot) -> LibrarySnapshot:
        """Return a changed snapshot without mutating the supplied library.

        Put creates or overwrites path. Replace requires an existing path and
        writes its Spec at new_path, relocating first when the paths differ.
        Rename relocates without evaluating children; delete does not cascade.
        Child library keys remain keys. The library owner reports KeyError for
        missing sources and ValueError for invalid families or destinations.
        Put/replace without a Spec raises InvalidInputError.
        """
        destination = self.new_path if self.new_path is not None else self.path
        if self.action == "delete":
            return library.deleted(self.path, kind=self.kind)
        if self.action == "rename":
            return library.renamed(self.path, destination, kind=self.kind)
        if self.action == "replace":
            library.entry_value(self.path, kind=self.kind)
            if destination != self.path:
                library = library.renamed(self.path, destination, kind=self.kind)
        if self.module is not None:
            return library.replaced_module(destination, self.module)
        if self.waveform is not None:
            return library.replaced_waveform(destination, self.waveform)
        raise InvalidInputError("Library put/replace requires a Spec")


@dataclass(frozen=True)
class EntryRequest:
    """Entry name: one ResultEntry-validated path segment, not a path or UUID."""

    name: str


@dataclass(frozen=True)
class PointFileSyncStatus:
    """Most recent external observation of one selected-point file.

    file is the installed owner version. disk_revision is the observed raw-byte
    SHA-256, or None for a missing file. state is pending when admission denies
    reload, reloaded after installation, or error after rejection. Only error
    has a nonempty error string, including the path and rejection reason.
    """

    file: PointFileRevision
    disk_revision: str | None
    state: Literal["pending", "reloaded", "error"]
    error: str | None


@dataclass(frozen=True)
class PointFileRevision:
    """One installed selected-point document, without reading disk.

    document is parameters, library or aliases, in that order in file lists.
    path is the absolute owner file. revision is the installed raw-byte SHA-256
    hex digest; None denotes a missing library or aliases file. Aliases may
    retain a content error, so their observed hash does not imply validity.
    """

    document: Literal["parameters", "library", "aliases"]
    path: Path
    revision: str | None


@dataclass(frozen=True)
class EntryInfo:
    """Opened entry directory name and absolute owner-supplied roots.

    name is its single-segment directory name in the configured Result root.
    result_path and database_path are the exact paths owned by this entry.
    """

    name: str
    result_path: Path
    database_path: Path


@dataclass(frozen=True)
class WorkingPointInfo:
    """Point metadata and the last parameter-only availability observation.

    label is the persisted directory label; description is general.description,
    empty when absent or unreadable. available means its parameters passed the
    owner schema probe, not that library/aliases are usable. error is None on
    success, otherwise the failed document path and diagnostic.
    """

    label: str
    description: str
    available: bool = True
    error: str | None = None


@dataclass(frozen=True)
class PointUseRequest:
    """Select label, a safe single path segment, in the currently opened entry."""

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
    """Create and select a saved point in the opened entry.

    label must satisfy the entry owner's safe single path segment rule.
    Context new_point rejects an existing destination with InvalidInputError and
    the entry owner's exists message before writes. It rejects an unsafe label
    with InvalidInputError before any source reads or writes, including when
    clone_from is supplied.
    description replaces the seed/clone description, including empty text.
    origin attributes the resource command. clone_from is a same-entry label;
    None copies setup components and uses an empty library.
    """

    label: str
    description: str
    origin: CommandOrigin
    clone_from: str | None = None


class PointBindingError(RuntimeError):
    """A point was created but could not be installed; never delete or retry it.

    entry_name identifies the entry within the current result root; point is
    the retained label. created is always True. The raising service chains the
    installation cause; Open/Use is recovery, not repeating New.
    """

    entry_name: str
    point: str
    created: Literal[True] = True

    def __init__(self, *, entry_name: str, point: str) -> None:
        self.entry_name = entry_name
        self.point = point
        super().__init__(
            f"Point {point!r} in entry {entry_name!r} was created but could not be "
            "installed. Use the same point to recover; do not repeat New."
        )


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
