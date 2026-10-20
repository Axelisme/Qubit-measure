"""Per-value sources shared by setup and complete working points."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from pydantic import BaseModel, ConfigDict

from .ledger import RecordsLedger


class ClonedFrom(TypedDict):
    """Direct clone origin: entry_id is its UUID, point its safe point label."""

    entry_id: str
    point: str


@dataclass(frozen=True)
class Provenance:
    """A value's source; stderr uses the view's working units.

    source is manual or an event id in this entry's ledger. at preserves the
    caller's timestamp; ordinary writes use UTC. A copied value may identify
    its direct source entry and point, even after cross-entry import.
    Each meta query returns independent clone metadata.
    """

    source: str
    kind: str | None
    run_id: str | None
    at: str
    stderr: float | None
    cloned_from: ClonedFrom | None = None


class ProvenanceData(BaseModel):
    """Persistent source metadata, in the value's working units.

    source is manual or an event ID in the owning entry's ledger. kind names
    the analysis or is None; run_id is the optional run identity. at preserves
    the source timestamp. stderr is optional uncertainty in working units.
    cloned_from identifies the direct source entry UUID and safe point label,
    or None for a value accepted here. Unknown fields are rejected.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    source: str
    kind: str | None
    run_id: str | None
    at: str
    stderr: float | None
    cloned_from: ClonedFrom | None = None

    @classmethod
    def from_provenance(cls, source: Provenance) -> ProvenanceData:
        """Validate and copy a public source record without consulting its ledger."""
        return cls(
            source=source.source,
            kind=source.kind,
            run_id=source.run_id,
            at=source.at,
            stderr=source.stderr,
            cloned_from=deepcopy(source.cloned_from),
        )

    def to_provenance(self) -> Provenance:
        """Copy all persisted fields into the public working-unit source record."""
        return Provenance(
            self.source,
            self.kind,
            self.run_id,
            self.at,
            self.stderr,
            deepcopy(self.cloned_from),
        )


class MissingSourceError(ValueError):
    """A well-addressed provenance source names no event in its local ledger.

    Ledger schema, attachment and I/O failures are not missing-source errors.
    """


def validate_source(provenance: Provenance, ledger: Path, entry_id: str) -> None:
    """Validate a manual or local event source without producing an event.

    ledger is this entry's ledger.jsonl path, entry_id its UUID. Unknown sources
    raise MissingSourceError with their KeyError cause; schema, attachment and I/O errors
    from RecordsLedger.get propagate. Manual sources do not access the ledger.
    """
    validate_sources((provenance,), ledger, entry_id)


def validate_sources(
    provenances: Iterable[Provenance], ledger: Path, entry_id: str
) -> None:
    """Validate all local sources with one ledger read, preserving input order.

    ledger is this entry's ledger.jsonl path, entry_id its UUID. Manual sources
    need no I/O. Duplicate event IDs and attachments are checked once per call.
    Missing IDs raise MissingSourceError with the original KeyError cause;
    ledger schema, attachment and I/O failures propagate unchanged.
    """
    sources = tuple(
        dict.fromkeys(p.source for p in provenances if p.source != "manual")
    )
    if not sources:
        return
    try:
        RecordsLedger(ledger.parent, entry_id=entry_id).get_many(sources)
    except KeyError as exc:
        raise MissingSourceError(
            f"{ledger}: source {exc.args[0]!r} does not exist"
        ) from exc
