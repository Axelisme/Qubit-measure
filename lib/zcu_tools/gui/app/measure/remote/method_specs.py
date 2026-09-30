"""Qt-free public remote method contract projection."""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import MethodSpec

from .method_entries import METHOD_ENTRIES
from .method_entries._registry import build_method_specs

METHOD_SPECS: dict[str, MethodSpec] = build_method_specs(METHOD_ENTRIES)
