"""Fixed Qt presentation for the new manual samples action."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qtpy.QtWidgets import QLabel, QPushButton
from zcu_tools.gui.app.workflows.samples import SampleExportReport

if TYPE_CHECKING:
    from .window import T1WorkflowWindow

logger = logging.getLogger(__name__)


def set_samples_action(
    button: QPushButton, *, eligible: bool, reason: str | None
) -> None:
    """Display the host's decided eligibility/reason on the Qt owner.

    button is the Window's samples control; eligible means terminal/admitted.
    reason is the captured/current attachment's unsupported explanation or None.
    Disables unsupported actions and exposes their reason as a tooltip. This
    function performs no domain classification, capture, I/O or run mutation.
    """
    button.setEnabled(eligible and reason is None)
    button.setToolTip(reason or "Explicitly append successful flux points to samples")


def show_samples_report(message: QLabel, report: SampleExportReport) -> None:
    """Show this call's successful row count and exact owner path on the Qt owner.

    message is the Window's diagnostic label. report is the host's successful
    append receipt, not a native measurement result. No domain state changes.
    """
    message.setText(f"Samples: {report.written_count} rows, {report.path}")


def invoke_samples(window: T1WorkflowWindow, message: QLabel) -> None:
    """Invoke the Window's common callable for its explicit run target on Qt owner.

    window supplies the retained Controller target and public export_samples.
    No attached target is a no-op. Successful export already displays its receipt;
    failures are isolated to message and logged with their original chained cause.
    No retry, alternate target, rollback or lifecycle change is performed here.
    """
    run_id = window.controller.run_id
    if run_id is None:
        return
    try:
        window.export_samples(run_id)
    except Exception as error:
        message.setText(str(error))
        logger.exception("Workflow samples action failed")
