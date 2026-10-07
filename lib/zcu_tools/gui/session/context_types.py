"""Typed parameter destinations and detached source diagnostics."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict

from zcu_tools.gui.expected_error import FailedPreconditionError


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
