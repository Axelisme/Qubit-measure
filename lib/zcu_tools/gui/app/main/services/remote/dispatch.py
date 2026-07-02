"""Runtime method registry projection for RemoteControlAdapter."""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import BoundMethod

from .method_entries import METHOD_ENTRIES
from .method_entries._registry import build_dispatch_registry

METHOD_REGISTRY: dict[str, BoundMethod] = build_dispatch_registry(METHOD_ENTRIES)
