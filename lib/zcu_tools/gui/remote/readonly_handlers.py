"""Generic read-only RPC handlers shared by the observe-only GUI apps.

Fluxdep and dispersive expose dispatch tables of pure
``(adapter, params) -> wire dict`` handlers. Shared handlers only touch
``ctrl.state``: resource versions read its VersionTable, and project info uses
``gui.project.project_info_payload``.

These take an ``adapter`` typed loosely (``Any``) because they only reach the
common ``adapter.ctrl.state`` path; each app registers them under its own dotted
method name in its dispatch table. App-specific handlers such as ``state.check``
stay in the app's ``dispatch.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from zcu_tools.gui.project import project_info_payload


def h_resources_versions(
    adapter: Any, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Snapshot the app's resource version table (mcp<->RPC bookkeeping only)."""
    del params
    return {"versions": adapter.ctrl.state.version.snapshot()}


def h_project_info(adapter: Any, params: Mapping[str, object]) -> Mapping[str, object]:
    """Project identity for apps using the shared ``gui.project.ProjectInfo``."""
    del params
    return project_info_payload(adapter.ctrl.state.project)


__all__ = ["h_project_info", "h_resources_versions"]
