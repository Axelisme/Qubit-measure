"""Generic read-only RPC handlers shared by the observe-only GUI apps.

The three read-only apps (``fluxdep`` / ``dispersive`` / ``autofluxdep``) each
expose a dispatch table of pure ``(adapter, params) -> wire dict`` handlers. A few
of those are byte-identical across apps because they only touch the shared
``ctrl.state`` surface:

  - :func:`h_resources_versions` reads ``ctrl.state.version`` (every app's State
    carries the same VersionTable) — shared by all three.
  - :func:`h_project_info` returns the shared :func:`gui.project.project_info_payload`
    for ``ctrl.state.project`` — shared by ``fluxdep`` / ``dispersive`` only.
    ``autofluxdep``'s ProjectInfo has a different shape (``params_path``, nullable),
    so it keeps an app-local handler.

These take an ``adapter`` typed loosely (``Any``) because they only reach the
common ``adapter.ctrl.state`` path; each app registers them under its own dotted
method name in its dispatch table. App-specific handlers (every ``state.check``,
autofluxdep's ``project.info``) stay in the app's ``dispatch.py``.
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
