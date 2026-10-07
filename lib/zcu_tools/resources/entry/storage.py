"""Bound point capabilities for run persistence."""

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from .ledger import RecordsLedger
from .points import PointView


@dataclass(frozen=True)
class PointStorageBinding:
    """One point's cached context and exact owning persistence capabilities.

    entry_id is the immutable owning UUID; entry_name is its captured name.
    point is the bound label and description is its cached text or None.
    result_path and database_path are absolute entry directories, not roots.
    ledger is a RecordsLedger bound to that entry's existing records directory.
    """

    entry_id: UUID
    entry_name: str
    point: str
    description: str | None
    result_path: Path
    database_path: Path
    ledger: RecordsLedger


def capture_point_storage(view: PointView) -> PointStorageBinding:
    """Capture identity and description from this bound point without reload.

    Use its constructor origin and cached snapshot, never cwd or GUI focus.
    Do not reopen ResultEntry, refresh YAML, create outputs or append events.
    RecordsLedger checks the existing records directory; its I/O errors and
    invalid bound UUID propagate. No hardware is read.
    """
    raise NotImplementedError
