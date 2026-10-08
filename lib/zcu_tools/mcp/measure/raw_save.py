"""Typed captures of synchronous native Save reports for MCP callers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from pydantic import TypeAdapter, ValidationError

from zcu_tools.mcp.measure.session import GuiConnection, GuiRpcError


@dataclass(frozen=True)
class SaveOutput:
    """One destination from the native owner's latest attempt.

    format is data_h5, labber, png or json. path is an absolute local path.
    status is reserved, saved or failed; reserved does not confirm a write.
    error is the owner's failure diagnostic, or None for other outputs.
    """

    format: Literal["data_h5", "labber", "png", "json"]
    path: str
    status: Literal["reserved", "saved", "failed"]
    error: str | None = None


@dataclass(frozen=True)
class SaveArtifact:
    """One whole native artifact, without a second persistence policy.

    status is the owner's aggregate: Not saved, Saved, Partially saved or
    Unsaved changes. members maps data, figure:<name> or analysis:<pane> to
    ordered outputs. Member kinds remain distinct even when names coincide.
    """

    status: Literal["Not saved", "Saved", "Partially saved", "Unsaved changes"]
    members: dict[str, list[SaveOutput]]


@dataclass(frozen=True)
class SaveReport:
    """The complete synchronous tab.save_data/tab.save_artifacts wire reply.

    run_id is confirmed by the GUI after a persistent output succeeds, or None
    before that. A failed resave may still carry the confirmed run_id.
    artifacts maps opaque sections to opaque artifact names and all their
    members/outputs from this attempt; an empty map means no attempted capture.
    Reports do not accumulate successful prefixes from earlier saves.
    Paths describe local outputs, not a guarantee that the files still exist.
    """

    run_id: str | None
    artifacts: dict[str, dict[str, SaveArtifact]]


@dataclass(frozen=True)
class RawSaveRequest:
    """A fixed native source and caller-owned dispatch admission.

    tab is the GUI locator. run_op is the opaque session Run handle whose data
    must be saved. before_start is an optional nonblocking callback under the RPC
    lock; it may reject by raising, but must not block or send RPCs.
    """

    tab: str
    run_op: int
    before_start: Callable[[], None] | None = None


def save_raw_data(
    connection: GuiConnection,
    request: RawSaveRequest,
    *,
    observe: Callable[[SaveReport | None], None],
) -> SaveReport:
    """Save the fixed Run once and capture the complete synchronous report.

    observe receives None immediately before an admitted dispatch, replacing
    the previous attempt, then the validated report when it returns. It must
    not block or send RPCs. request.before_start rejection propagates without
    changing the previous capture. Native per-output failures stay in the
    returned report; transport/admission failures propagate, and malformed
    wire reports raise GuiRpcError(reason="incompatible_wire").
    There is no Save handle, await, snapshot read, retry, reconnect or rollback.
    Run operation handles retain their existing session-local meaning.
    """

    def admit() -> None:
        if request.before_start is not None:
            request.before_start()
        observe(None)

    reply = connection.send_gui_rpc(
        "tab.save_data",
        {"tab_id": request.tab},
        run_operation_handle=request.run_op,
        before_send=admit,
    )
    try:
        report = TypeAdapter(SaveReport).validate_python(reply)
    except ValidationError as error:
        raise GuiRpcError(
            "Invalid native Save report", reason="incompatible_wire"
        ) from error
    observe(report)
    return report
