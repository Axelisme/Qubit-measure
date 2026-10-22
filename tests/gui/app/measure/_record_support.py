"""Narrow record envelopes published through the GUI State seam."""

from typing import Any

from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.state import RunPaneState


def require_record(pane: RunPaneState) -> RunRecord[Any, Any]:
    """Return the published record, failing when the pane has no result."""
    assert pane.result is not None, "Run did not publish a record"
    return pane.result
