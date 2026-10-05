"""Raw-save completion and confirmed-prefix receipts for recipe callers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from threading import Condition, Event
from typing import Literal

from zcu_tools.mcp.measure.operation_wait import await_operation
from zcu_tools.mcp.measure.session import GuiConnection, GuiRpcError


@dataclass(frozen=True)
class RawSaveReceipt:
    """Capture of a raw save, retaining any previously confirmed saved prefix.

    status is not_started, saving, saved, failed or unknown. reserved_path is the
    latest planned file path; path is a confirmed saved path, possibly from an
    earlier successful save. None means not captured. operation_outcome retains
    the latest native terminal reply, or None before capture. Native fields are
    diagnostic facts, not a second operation policy.
    """

    status: Literal["not_started", "saving", "saved", "failed", "unknown"] = (
        "not_started"
    )
    reserved_path: str | None = None
    path: str | None = None
    operation_outcome: Mapping[str, object] | None = None


@dataclass(frozen=True)
class RawSaveRequest:
    """A native save source and its caller-owned dispatch admission.

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
    closed: Event,
    condition: Condition,
    previous: RawSaveReceipt,
    observe: Callable[[int | None, RawSaveReceipt], None],
) -> RawSaveReceipt:
    """Start and await one raw save on its fixed Run source in this worker.

    request binds the GUI locator, opaque Run handle and dispatch admission.
    closed/condition belong to the owning execution, as in await_operation. previous
    retains earlier confirmed saves. observe receives each new receipt plus the
    admitted save handle, or None if no handle was received; it must not send RPCs.
    Rejection by request.before_start propagates without changing the receipt.

    Once admitted, cancellation does not interrupt the save. Native failures and
    delivery/close errors propagate after observe captures failed/unknown facts.
    Handler timeouts and response encoding failures retain unknown outcomes.
    Reserved paths do not become confirmed until the true finished outcome. No
    retry, reconnect, rollback, or worker creation occurs.
    """
    receipt = replace(previous, operation_outcome=None)
    save_op: int | None = None
    admission_failed = False
    failure_observed = False

    def admit() -> None:
        nonlocal admission_failed
        if request.before_start is not None:
            try:
                request.before_start()
            except Exception:  # Preserve the caller's admission error.
                admission_failed = True
                raise

    try:
        started = connection.send_gui_rpc(
            "tab.save_data",
            {"tab_id": request.tab},
            run_operation_handle=request.run_op,
            before_send=admit,
        )
        save_op = started["handle"]
        path = started["data_path"]
        receipt = replace(
            previous, status="saving", reserved_path=path, operation_outcome=None
        )
        observe(save_op, receipt)
        completion = await_operation(
            connection, started["handle"], closed=closed, condition=condition
        )
        if completion.status != "finished":
            receipt = replace(
                receipt, status="failed", operation_outcome=completion.native
            )
            observe(save_op, receipt)
            failure_observed = True
            raise GuiRpcError(
                str(completion.native.get("error", "Raw save failed")),
                reason="raw_save_failed",
            )
        receipt = RawSaveReceipt("saved", path, path, completion.native)
        observe(save_op, receipt)
        return receipt
    except Exception as error:  # Capture failure facts, then propagate.
        if not admission_failed and not failure_observed:
            reason = error.reason if isinstance(error, GuiRpcError) else None
            unknown = reason in (
                "gui_transport_timeout",
                "gui_handler_timeout",
                "response_encoding_failed",
                "connection_lost",
                "message_too_large",
            ) or (reason == "session_closed" and receipt.status == "saving")
            observe(
                save_op, replace(receipt, status="unknown" if unknown else "failed")
            )
        raise
