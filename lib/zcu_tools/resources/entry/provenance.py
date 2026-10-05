"""Per-value sources shared by setup and complete working points."""

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

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


def validate_source(provenance: Provenance, ledger: Path, entry_id: str) -> None:
    """Validate a manual or local event source without producing an event.

    ledger is this entry's ledger.jsonl path, entry_id its UUID. Unknown sources
    raise ValueError with their KeyError cause; schema, attachment and I/O errors
    from RecordsLedger.get propagate. Manual sources do not access the ledger.
    """
    if provenance.source == "manual":
        return
    try:
        RecordsLedger(ledger.parent, entry_id=entry_id).get(provenance.source)
    except KeyError as exc:
        raise ValueError(
            f"{ledger}: source {provenance.source!r} does not exist"
        ) from exc
