"""Present input-restore outcomes without clearing unrelated diagnostics."""

from qtpy.QtWidgets import QLabel

from zcu_tools.gui.session.persistence import PersistenceError

from .persistence import WorkflowInputRestoreReport


class WorkflowInputRestoreMessage:
    """Track restore-message ownership for one Window diagnostics label.

    Construct and use on the GUI owner thread. The Window retains the label
    and this presenter for the same lifetime; this object does not delete it.
    Other diagnostics writers may use the label directly. Only restore text
    formatting and recognition of the previous restore message belong here.
    """

    def __init__(self, label: QLabel) -> None:
        """Use the existing diagnostics label without changing its current text."""
        self._label = label
        self._previous = ""

    def show(
        self,
        report: WorkflowInputRestoreReport,
        load_error: PersistenceError | None = None,
    ) -> None:
        """Display a completed restore outcome without dialogs, form edits or I/O.

        report lists discarded plan/knobs paths in display order. load_error is
        the caretaker read/decode failure that selected defaults, or None.
        Nonempty outcomes replace the label text. Empty successful outcomes
        clear it only when it still matches this presenter's previous message,
        preserving intervening run or command diagnostics. No failures are caught.
        """
        messages = [str(load_error)] if load_error is not None else []
        for form, paths in (
            ("plan", report.plan_dropped),
            ("knobs", report.knobs_dropped),
        ):
            messages.extend(
                f"Discarded {form}: {'.'.join(str(part) for part in path)}"
                for path in paths
            )
        text = "; ".join(messages)
        if text or self._label.text() == self._previous:
            self._label.setText(text)
        self._previous = text
