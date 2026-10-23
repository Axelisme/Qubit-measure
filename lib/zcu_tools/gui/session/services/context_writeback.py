"""Bind selected-point commands to the ordered writeback execution port."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    from zcu_tools.format_version import YamlMap, YamlValue
    from zcu_tools.gui.session.context_types import SavedPoint
    from zcu_tools.gui.session.services.context import ContextService
    from zcu_tools.resources.entry import ResultEntry
    from zcu_tools.resources.entry.ledger_models import JsonObject
    from zcu_tools.resources.entry.writeback import LibraryUpdate, ParameterUpdate


class ContextWritebackTarget:
    """Bind a captured saved point to resources.entry.writeback.WritebackTarget.

    context is the owner-thread service used for parameter/library writes.
    target fixes the entry UUID and safe point label; entry must be that
    target's open ResultEntry. Binding stores these handles without reading or
    writing files. It neither follows later selection nor creates an entry.
    Caller must supply matching handles; binding does not validate them.

    Public fields describe the bound destination: entry_id is entry's UUID
    string, ledger is its RecordsLedger for accepted events, and point is the
    captured nonempty point label. Treat these fields as read-only.

    set_parameters/set_modules implement the nonempty ordered batch contract
    of WritebackTarget. Each commits one file and returns one detached actual
    value per update in input order, only after the entire batch succeeds.
    Values reflect each validated draft update before later overwrites.
    ContextService owns target admission, schema/source/conflict checks and
    publication. Missing selection, cross-entry targets, invalid data, conflicts
    and storage errors propagate; a failed batch writes no items and publishes
    nothing. Same-entry nonselected destinations stay unselected. Methods must
    run on context's owner thread; thread violations propagate.

    append_sample delegates validation and append-only storage to the bound
    entry's JsonlSampleTable. No method appends an accepted event itself or
    provides a transaction across parameter, library and sample files.
    """

    def __init__(
        self, context: ContextService, target: SavedPoint, entry: ResultEntry
    ) -> None:
        """Bind matching context/target/entry handles as described by the class."""
        self._context = context
        self._target = target
        self._entry = entry
        self.entry_id = entry.entry_id
        self.ledger = entry.ledger
        self.point = target.point

    def set_parameters(
        self, updates: tuple[ParameterUpdate, ...]
    ) -> tuple[YamlValue, ...]:
        """Commit nonempty updates to the fixed point under the class contract.

        ParameterUpdate supplies concrete dotted paths, YAML values and optional
        metadata replacements. Return each update's validated value; owner
        validation, target, conflict and I/O failures propagate without a commit.
        """
        return self._context.set_parameters(self._target, updates)

    def set_modules(self, updates: tuple[LibraryUpdate, ...]) -> tuple[YamlMap, ...]:
        """Commit nonempty library updates under the class batch contract.

        LibraryUpdate supplies each concrete key, named module/waveform Spec
        and optional provenance. Return each validated Spec mapping; owner
        target, schema, source, conflict and I/O failures propagate without a commit.
        """
        return self._context.set_modules(self._target, updates)

    def append_sample(
        self,
        component: str,
        fields: JsonObject,
        *,
        run_id: str,
        at: datetime,
        source: str,
    ) -> JsonObject:
        """Append a sample to component and return its detached complete JSON row.

        component is a safe single path segment. fields contains finite JSON
        measurements in caller units, without the reserved run_id, point, at or
        source keys. run_id is a nonempty acquisition identifier, at is the
        accepted event's timezone-aware UTC datetime, and source is its nonempty
        event ID. The row records this target's point label. Inputs stay unchanged.
        Invalid paths/metadata/JSON and I/O errors propagate. Existing rows stay
        unchanged, including earlier rows with the same run_id. An append is not
        transactional with other files and an I/O failure may leave a partial row.
        """
        return self._entry.samples(component).append(
            run_id=run_id, point=self.point, at=at, source=source, **fields
        )
