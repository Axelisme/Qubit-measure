"""Bound point capabilities for run persistence."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from uuid import UUID

from ._path_segments import validate_path_segment
from .ledger import RecordsLedger
from .points import PointView
from .roles import RoleView, capture_role_components
from .views import point_origin, snapshot


@dataclass(frozen=True)
class PointStorageBinding:
    """One point's cached context and exact owning persistence capabilities.

    entry_id is the immutable owning UUID; entry_name is its captured name.
    point is the bound label and description is its cached text or None.
    result_path and database_path are absolute entry directories, not roots.
    ledger is a RecordsLedger bound to that entry's existing records directory.
    roles is an isolated, read-only mapping of captured roles to concrete names;
    it is empty when capture_point_storage was called without a RoleView.
    """

    entry_id: UUID
    entry_name: str
    point: str
    description: str | None
    result_path: Path
    database_path: Path
    ledger: RecordsLedger
    roles: Mapping[str, str]


def capture_point_storage(
    view: PointView, *, roles: RoleView | None = None
) -> PointStorageBinding:
    """Capture this bound point's identity, description and optional fixed roles.

    roles must come from PointView.resolve on this same entry/point with the
    same owning roots. Another handle of that point is accepted; other origins
    raise ValueError, including for empty roles. None captures an empty mapping.
    Names are copied without re-resolving focus, references or changed values.
    Use its constructor origin and cached snapshot, never cwd or GUI focus.
    Do not reopen ResultEntry, refresh YAML, create outputs or append events.
    RecordsLedger checks the existing records directory; its I/O errors and
    invalid bound UUID propagate. No hardware is read.
    """
    origin = point_origin(view)
    entry_id = UUID(origin.entry_id)
    names = (
        MappingProxyType({})
        if roles is None
        else capture_role_components(roles, origin)
    )
    relative = origin.source.resolve().relative_to(origin.result_root)
    if (
        len(relative.parts) != 4
        or relative.parts[1] != "points"
        or relative.parts[3] != "point.yaml"
    ):
        raise ValueError("Point origin must be <entry>/points/<point>/point.yaml")
    entry_name, _, point, _ = relative.parts
    validate_path_segment(entry_name, field="entry_name")
    validate_path_segment(point, field="point")
    document = snapshot(view)
    result_path = origin.result_root / entry_name
    database_path = origin.database_root / entry_name
    return PointStorageBinding(
        entry_id=entry_id,
        entry_name=entry_name,
        point=point,
        description=document.general.description,
        result_path=result_path,
        database_path=database_path,
        ledger=RecordsLedger(result_path / "records", entry_id=str(entry_id)),
        roles=names,
    )
