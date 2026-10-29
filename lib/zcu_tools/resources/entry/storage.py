"""Bound point capabilities for run persistence."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType

from zcu_tools.resources.document_names import PARAMETERS_FILENAME
from zcu_tools.resources.record_table import JsonlRecordTable

from ._path_segments import validate_path_segment
from .aliases import AliasValue
from .ledger import RecordsLedger
from .points import PointView
from .roles import RoleView, capture_role_components
from .save_layout import SaveLayout
from .views import point_origin


@dataclass(frozen=True)
class PointStorageBinding:
    """One point's cached context and exact owning persistence capabilities.

    entry_name locates the entry in its result root.
    point is the bound label and description is its cached text or None.
    result_path and database_path are absolute entry directories, not roots.
    ledger is a RecordsLedger bound to that entry's records directory, created on first append if absent.
    roles is a detached mapping of captured aliases to scalar names or ordered lists;
    it is empty when capture_point_storage was called without a RoleView.
    """

    entry_name: str
    point: str
    description: str | None
    result_path: Path
    database_path: Path
    ledger: RecordsLedger
    roles: Mapping[str, AliasValue]


@dataclass(frozen=True)
class RunOutputBinding:
    """Recorded output names under current roots, independent of live points.

    result_path is the absolute Result entry output directory. database_path
    is its absolute Database entry output directory. Neither must exist.
    point is the safe recorded label, or unnamed_point when it was absent.
    Use bind_run_outputs to validate names and root containment. Binding and
    layout planning do not read parameter documents or create files.
    """

    result_path: Path
    database_path: Path
    point: str

    def layout(self, *, stem: str, saved_at: datetime) -> SaveLayout:
        """Plan outputs for a safe run stem and timezone-aware Save timestamp.

        Return a SaveLayout without creating files. Unsafe path segments,
        escaping symlinks or a naive saved_at raise ValueError.
        """
        return SaveLayout(
            result_path=self.result_path,
            database_path=self.database_path,
            point=self.point,
            stem=stem,
            saved_at=saved_at,
        )

    def prepare_ledger(self) -> RecordsLedger:
        """Create the Result output directory and bind its append-only ledger.

        No setup or point is opened or manufactured. Reject a records symlink
        escaping this entry before creating directories or appending evidence.
        """
        records = self.result_path / "records"
        if records.resolve().parent != self.result_path.resolve():
            raise ValueError(f"{records}: records escape their entry")
        self.result_path.mkdir(parents=True, exist_ok=True)
        return RecordsLedger(records)


def bind_run_outputs(
    *,
    result_root: Path,
    database_root: Path,
    entry_name: str | None,
    point: str | None,
) -> RunOutputBinding:
    """Bind current roots plus recorded names without locating a live entry.

    None names use unnamed_entry and unnamed_point. Explicit names must be
    nonempty safe path segments; neither entry path may escape its owning root.
    No parameter, metadata, ledger or original data file is read.
    """
    name = "unnamed_entry" if entry_name is None else entry_name
    label = "unnamed_point" if point is None else point
    validate_path_segment(name, field="entry_name")
    validate_path_segment(label, field="point")
    result_path = result_root / name
    database_path = database_root / name
    for root, path in ((result_root, result_path), (database_root, database_path)):
        if path.resolve().parent != root.resolve():
            raise ValueError(f"{path}: entry escapes its output root")
    return RunOutputBinding(result_path.resolve(), database_path.resolve(), label)


def bind_record_table(result_path: Path, component: str) -> JsonlRecordTable:
    """Bind entry record history without reading schemas, ledger or files.

    result_path is the exact entry Result directory, not its parent root.
    component is a safe single path segment, never a role alias, and is bound
    into every appended row's component envelope field. Rows live at
    records/components/<component>.jsonl. Unsafe names or symlinks escaping the entry raise
    ValueError. The returned owner handles append, latest-row read and export.
    """
    validate_path_segment(component, field="component")
    path = result_path / "records" / "components" / f"{component}.jsonl"
    record_root = result_path.resolve() / "records" / "components"
    if not path.resolve().is_relative_to(record_root):
        raise ValueError(f"{path}: records escape their component namespace")
    return JsonlRecordTable(path, component=component)


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
    RecordsLedger permits a missing records directory; other I/O errors and
    invalid source locations propagate. No hardware is read.
    """
    origin = point_origin(view)
    names = (
        MappingProxyType({})
        if roles is None
        else capture_role_components(roles, origin)
    )
    relative = origin.source.resolve().relative_to(origin.result_root)
    if (
        len(relative.parts) != 4
        or relative.parts[1] != "points"
        or relative.parts[3] != PARAMETERS_FILENAME
    ):
        raise ValueError(
            f"Point origin must be <entry>/points/<point>/{PARAMETERS_FILENAME}"
        )
    entry_name, _, point, _ = relative.parts
    validate_path_segment(entry_name, field="entry_name")
    validate_path_segment(point, field="point")
    result_path = origin.result_root / entry_name
    database_path = origin.database_root / entry_name
    return PointStorageBinding(
        entry_name=entry_name,
        point=point,
        description=view.description,
        result_path=result_path,
        database_path=database_path,
        ledger=RecordsLedger(result_path / "records"),
        roles=names,
    )
