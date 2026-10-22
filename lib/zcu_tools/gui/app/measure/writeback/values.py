"""Fixed-destination reads shared by draft preview and retained receipts."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.context_control import ContextControlPort
from zcu_tools.gui.session.types import ParameterContext
from zcu_tools.resources.entry import document_value

from .contracts import WritebackApplied
from .items import ParameterWriteback, SampleWriteback, WritebackItem

if TYPE_CHECKING:
    from .service import WritebackDraft


def destination_path(item: WritebackItem, parameters: ParameterContext) -> str:
    """Resolve the concrete destination from the fixed draft capture."""
    if isinstance(item, SampleWriteback):
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
    """Read a fixed destination, including accepted Sample row metadata.

    current is the latest fixed-target context, never active focus. path must
    already be concrete. Samples read the draft's run-specific row, not a module.
    Missing captured parameters or I/O errors propagate.
    """
    if isinstance(item, SampleWriteback) and draft.run_id is not None:
        parameters = draft.parameters
        if parameters is None:
            raise FailedPreconditionError("Writeback requires captured parameters")
        return write_port.read_sample(parameters.target, path, draft.run_id)
    return destination_value(item, current, path)


def applied_values(
    write_port: ContextControlPort,
    draft: WritebackDraft,
    items: tuple[WritebackItem, ...],
) -> tuple[WritebackApplied, ...]:
    """Detach confirmed items at their fixed destination before yielding the owner.

    items must be the confirmed successful prefix of the current submission.
    Empty items need no context. Context/read failures propagate, never substitute
    proposal values for actual stored values.
    """
    if not items:
        return ()
    parameters = draft.parameters
    if parameters is None:
        raise FailedPreconditionError("Writeback requires captured parameters")
    current = write_port.read_target(parameters.target, roles=parameters.roles)
    return tuple(
        WritebackApplied(
            id=item.session_id,
            kind="sample"
            if isinstance(item, SampleWriteback)
            else "parameter"
            if isinstance(item, ParameterWriteback)
            else "module",
            target=destination_path(item, parameters),
            value=deepcopy(
                read_destination_value(
                    write_port, draft, item, current, destination_path(item, parameters)
                )
            ),
            destination={
                "kind": "saved_point",
                "entry_id": parameters.target.entry_id,
                "point": parameters.target.point,
            },
        )
        for item in items
    )
