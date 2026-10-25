"""Fixed-destination preview reads and confirmed owner receipt projection."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.context_control import ContextControlPort
from zcu_tools.gui.session.types import ParameterContext
from zcu_tools.resources.entry import document_value
from zcu_tools.resources.entry.writeback import WritebackResult

from .contracts import WritebackApplied
from .items import ParameterWriteback, RecordWriteback, WritebackItem

if TYPE_CHECKING:
    from .service import WritebackDraft


def destination_path(item: WritebackItem, parameters: ParameterContext) -> str:
    """Resolve the concrete destination from the fixed draft capture."""
    if isinstance(item, RecordWriteback):
        return item.target_name
    if isinstance(item, ParameterWriteback):
        return parameters.resolve_path(item.target_name)
    return parameters.library_key(item.target_name)


def destination_value(
    item: WritebackItem, parameters: ParameterContext, path: str
) -> object:
    """Read a concrete parameter path or raw library entry; absent means None."""
    if isinstance(item, ParameterWriteback):
        # The destination is already concrete; do not resolve captured roles again.
        try:
            return document_value(parameters.document(), path)
        except AttributeError:
            return None
    library = parameters.library
    if path not in library.module_keys():
        return None
    return library.entry_value(path, kind="module")


def read_destination_value(
    write_port: ContextControlPort,
    draft: WritebackDraft,
    item: WritebackItem,
    current: ParameterContext,
    path: str,
) -> object:
    """Read a fixed destination, including accepted Record row metadata.

    current is the latest fixed-target context, never active focus. path must
    already be concrete. Records read the draft's run-specific row, not a module.
    Missing captured parameters or I/O errors propagate.
    """
    if isinstance(item, RecordWriteback) and draft.run_id is not None:
        parameters = draft.parameters
        if parameters is None:
            raise FailedPreconditionError("Writeback requires captured parameters")
        return write_port.read_record(parameters.target, path, draft.run_id)
    return destination_value(item, current, path)


def applied_values(
    draft: WritebackDraft,
    items: tuple[WritebackItem, ...],
    result: WritebackResult,
) -> tuple[WritebackApplied, ...]:
    """Associate the owner's ordered committed prefix with draft IDs/destination.

    items is the selected submission order; each committed index selects its
    item even when document grouping changes commit order. result contains
    actual values formed by each write owner. No destination read, source
    refresh or proposal-value substitution occurs. The captured SavedPoint supplies entry/point identity.
    """
    parameters = draft.parameters
    if parameters is None:
        raise FailedPreconditionError("Writeback requires captured parameters")
    values: list[WritebackApplied] = []
    for committed in result.committed:
        item = items[committed.index]
        values.append(
            WritebackApplied(
                id=item.session_id,
                kind="record"
                if isinstance(item, RecordWriteback)
                else "parameter"
                if isinstance(item, ParameterWriteback)
                else "module",
                target=committed.path,
                value=deepcopy(committed.value),
                destination={
                    "kind": "saved_point",
                    "entry_id": parameters.target.entry_id,
                    "point": parameters.target.point,
                },
            )
        )
    return tuple(values)
