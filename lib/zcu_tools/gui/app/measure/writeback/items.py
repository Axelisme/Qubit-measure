"""Mutable parameter, module and record proposals supplied by adapters."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from zcu_tools.core.element import ElementIdentity

if TYPE_CHECKING:
    from zcu_tools.cfg_model import ModuleSpec
    from zcu_tools.resources.entry.ledger_models import JsonObject


@dataclass
class WritebackItem(ABC):
    """Mutable proposal; stderr is its standard error in working units, or None.

    The destination is a direct container path or captured role path. selected
    sets the initial checkbox state (True by default); the draft owns later
    manual selection changes. note optionally explains the proposal or its
    initial selection in plain text; None means no note. Adapters decide quality,
    and editing or applying a draft does not reinterpret that note. Session
    identity belongs to the service, not the proposing adapter.
    element optionally carries the author's original index/name from the same
    Run/Load result; None means no element identity was supplied. It is display
    metadata only and never changes target_name, destination or write permission.
    """

    stderr: float | None = field(default=None, kw_only=True)
    element: ElementIdentity | None = field(default=None, kw_only=True)
    # ``target_name`` is the apply destination name (point parameter / library
    # module / record component). It is mutable: agent/UI
    # may retarget the writeback before applying.
    target_name: str
    description: str
    # ``session_id`` (``<kind>-<n>``, e.g. ``param-1``) is the stable identifier for
    # UI/wire/dedup. Stamped once by WritebackService when items are computed at
    # analyze time (never by the adapter), and decoupled from target_name so
    # retargeting does not change the id.
    session_id: str = field(default="", init=False)
    selected: bool = field(default=True, kw_only=True)
    note: str | None = field(default=None, kw_only=True)


@dataclass
class ParameterWriteback(WritebackItem):
    """Propose a point parameter value at target_name, resolved against the Run capture."""

    proposed_value: Any


@dataclass
class ModuleWriteback(WritebackItem):
    """Propose a persistent module Spec for the library root target_name.

    proposal is a concrete persistent ModuleSpec (pulse, readout or reset);
    child library keys stay keys. The writeback draft edits a Form copy of it,
    so the proposal object is never modified. role_id optionally names the
    role template this proposal represents while it is reviewed; the library
    does not persist it.
    """

    proposal: ModuleSpec
    role_id: str | None = None


@dataclass
class RecordWriteback(WritebackItem):
    """Append adapter-owned JSON record to target_name, a concrete component.

    record is detached and passed unchanged to the append owner, apart from its
    run_id/point/at/source provenance envelope. The framework does not interpret
    fields or select coordinate devices. Adapters document their shared format
    policy in lab authoring documentation. Review fields remain independent.
    IDs and wire receipt kinds use ``record``. The append owner stores rows in
    records/components/<target_name>.jsonl, separate from the event ledger.
    """

    record: JsonObject
