"""Read-only resource version RPC projection shared by GUI apps.

Each app registers this handler against its adapter.ctrl.state owner.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def h_resources_versions(
    adapter: Any, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Snapshot the app's resource version table (mcp<->RPC bookkeeping only)."""
    del params
    return {"versions": adapter.ctrl.state.version.snapshot()}


__all__ = ["h_resources_versions"]
