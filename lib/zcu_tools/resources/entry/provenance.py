"""Per-value sources shared by setup and complete working points."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from pydantic import TypeAdapter


class ClonedFrom(TypedDict):
    entry_id: str
    point: str


@dataclass(frozen=True)
class Provenance:
    """A value's source; stderr uses the view's working units.

    source is manual or an event id in this entry's ledger. at preserves the
    caller's timestamp; ordinary writes use UTC. A copied value may identify
    its direct same-entry point.
    Each meta query returns independent clone metadata.
    """

    source: str
    kind: str | None
    run_id: str | None
    at: str
    stderr: float | None
    cloned_from: ClonedFrom | None = None


def validate_source(provenance: Provenance, ledger: Path, entry_id: str) -> None:
    """Check a local reference, without interpreting or producing ledger events."""
    if provenance.source == "manual":
        return
    with ledger.open(encoding="utf-8") as stream:
        for line in stream:
            event = TypeAdapter(dict[str, object]).validate_python(json.loads(line))
            if event.get("id") == provenance.source:
                if event.get("entry_id") != entry_id:
                    raise ValueError(
                        f"{ledger}: source {provenance.source!r} belongs to another entry"
                    )
                return
    raise ValueError(f"{ledger}: source {provenance.source!r} does not exist")
