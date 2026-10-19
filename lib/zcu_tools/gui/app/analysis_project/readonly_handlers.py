"""Project identity RPC projection shared by the two analysis apps."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from zcu_tools.gui.app.analysis_project.project import project_info_payload


def h_project_info(adapter: Any, params: Mapping[str, object]) -> Mapping[str, object]:
    """Project identity for an analysis adapter holding ctrl.state.project."""
    del params
    return project_info_payload(adapter.ctrl.state.project)
