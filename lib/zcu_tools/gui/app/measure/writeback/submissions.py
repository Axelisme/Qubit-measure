"""GUI-lifetime submission identities and detached writeback receipts."""

from collections.abc import Generator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4

from zcu_tools.gui.expected_error import InvalidInputError

from .contracts import WritebackApplied


@dataclass(frozen=True)
class WritebackCompleted:
    """A confirmed item with its pane (analysis or post_analysis) identity."""

    subtab_id: Literal["analysis", "post_analysis"]
    item: WritebackApplied


@dataclass
class WritebackSubmission:
    """Detached submission observation, retained until the GUI owner is destroyed.

    submission_id is an opaque once-only ID, unrelated to cancellable operations.
    tab_id is the reservation's tab. pending means reserved or still executing,
    not that writes are absent. completed and failed are owner terminal outcomes.
    completed lists only confirmed writes; error is the first failure or None.
    """

    submission_id: str
    tab_id: str
    status: Literal["pending", "completed", "failed"] = "pending"
    completed: list[WritebackCompleted] = field(default_factory=list)
    error: str | None = None


class WritebackSubmissions:
    """Owner-thread registry shared by single-pane writes and captured acceptance.

    Reservation does not validate a draft, grant write permission or write data.
    A submission ID can be claimed only once and only for its original tab.
    Queries return detached in-memory observations without reading destinations.
    """

    def __init__(self) -> None:
        self._receipts: dict[str, WritebackSubmission] = {}
        self._submitted: set[str] = set()

    def begin(self, tab_id: str) -> WritebackSubmission:
        """Reserve a pending ID for the caller-validated tab; perform no writes."""
        receipt = WritebackSubmission(uuid4().hex, tab_id)
        self._receipts[receipt.submission_id] = receipt
        return deepcopy(receipt)

    def receipt(self, submission_id: str) -> WritebackSubmission:
        """Read a detached receipt; unknown/previous-GUI IDs raise InvalidInputError."""
        if submission_id not in self._receipts:
            raise InvalidInputError("unknown writeback submission_id")
        return deepcopy(self._receipts[submission_id])

    @contextmanager
    def submit(
        self, tab_id: str, submission_id: str | None
    ) -> Generator[WritebackSubmission | None]:
        """Claim once and record terminal outcome around synchronous owner work.

        None executes without retaining a submission receipt. A supplied ID must be reserved
        for tab_id and unused. The caller appends confirmed items before returning
        or raising; exceptions retain that prefix and mark the receipt failed.
        """
        if submission_id is None:
            yield None
            return
        observed = self.receipt(submission_id)
        if observed.tab_id != tab_id:
            raise InvalidInputError("writeback submission belongs to another tab")
        if submission_id in self._submitted:
            raise InvalidInputError("writeback submission already submitted")
        self._submitted.add(submission_id)
        receipt = self._receipts[submission_id]
        try:
            yield receipt
        except Exception as error:
            receipt.status = "failed"
            receipt.error = str(error)
            raise
        else:
            receipt.status = "failed" if receipt.error is not None else "completed"
