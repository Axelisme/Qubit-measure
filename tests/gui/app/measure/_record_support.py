"""Narrow record envelopes published through the GUI State seam."""

from typing import Any

from zcu_tools.experiment.metadata import RunMetadata
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.state import RunPaneState
from zcu_tools.gui.session.context_types import ContextTarget
from zcu_tools.gui.session.parameters import ParameterContext


def require_record(pane: RunPaneState) -> RunRecord[Any, Any]:
    """Return the published record, failing when the pane has no result."""
    assert pane.result is not None, "Run did not publish a record"
    return pane.result


def require_target(parameters: ParameterContext) -> ContextTarget:
    """Return a live persistent target, failing for stored/empty analysis content."""
    target = parameters.target
    assert target is not None, "Capture has no persistent target"
    return target


def require_metadata(pane: RunPaneState) -> RunMetadata:
    metadata = require_record(pane).metadata
    assert metadata is not None, "Run has no metadata"
    return metadata.project()
