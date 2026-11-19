"""Selected-point previews and confirmed owner receipt projection."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from zcu_tools.gui.session.context_control import ContextControlPort
from zcu_tools.gui.session.context_types import DocumentSetRef
from zcu_tools.gui.session.types import DocumentSetCapture
from zcu_tools.resources.entry import document_value
from zcu_tools.resources.entry import writeback as concrete

from .assets import arb_asset_current, concrete_arb_asset
from .contracts import WritebackApplied
from .items import (
    ArbAssetWriteback,
    ParameterWriteback,
    RecordWriteback,
    WritebackItem,
    writeback_kind,
)

if TYPE_CHECKING:
    from .service import WritebackDraft


def destination_path(item: WritebackItem, parameters: DocumentSetCapture) -> str:
    """Expand historical aliases/slots, leaving concrete destinations unchanged.

    parameters supplies only captured expression/library content, not a write
    target. Concrete parameter paths and module roots need no historical source;
    the destination owner validates them on read or application.
    """
    if isinstance(item, ArbAssetWriteback):
        return concrete_arb_asset(item).path
    if isinstance(item, RecordWriteback):
        return item.target_name
    if isinstance(item, ParameterWriteback):
        return (
            parameters.resolve_path(item.target_name)
            if item.target_name.startswith("$")
            else item.target_name
        )
    parts = item.target_name.split(".")
    if (
        len(parts) == 3
        and parts[1] in ("pulses", "readout", "reset")
        and not parts[0].startswith("$")
    ):
        return item.target_name
    return parameters.library_key(item.target_name)


def destination_value(
    item: WritebackItem, parameters: DocumentSetCapture, path: str
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
    current: DocumentSetCapture,
    path: str,
) -> object:
    """Read a fixed destination, including accepted Record row metadata.

    current is the active document set captured for this call. path must already
    be concrete. Records read the draft's run-specific row, not a module.
    Missing persistent targets and I/O errors propagate.
    """
    if isinstance(item, ArbAssetWriteback):
        return arb_asset_current(item, current)
    if isinstance(item, RecordWriteback) and draft.run_id is not None:
        target = current.target
        if target is None:
            raise ValueError("Current destination has no persistent target")
        return write_port.read_record(target, path, draft.run_id)
    return destination_value(item, current, path)


def applied_values(
    target: DocumentSetRef,
    items: tuple[WritebackItem, ...],
    result: concrete.WritebackResult,
) -> tuple[WritebackApplied, ...]:
    """Associate the owner's ordered committed prefix with draft IDs/destination.

    items is the selected submission order; each committed index selects its
    item even when document grouping changes commit order. result contains
    actual values formed by each write owner. No destination read, source
    refresh or proposal-value substitution occurs. target is the GUI-selected
    document set used for this submission and supplies the receipt's entry/point names.
    """
    values: list[WritebackApplied] = []
    for committed in result.committed:
        item = items[committed.index]
        values.append(
            WritebackApplied(
                id=item.session_id,
                kind=writeback_kind(item),
                target=committed.path,
                value=deepcopy(committed.value),
                destination={
                    "entry_name": target.entry_name,
                    "point": target.point,
                },
            )
        )
    return tuple(values)
