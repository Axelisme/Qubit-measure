"""Qt-free public remote method contract aggregator.

Grouped spec tables live in ``method_spec_groups/``. This module remains the
single public import path for MCP schema generation and dispatch binding.
"""

from __future__ import annotations

from collections.abc import Mapping

from zcu_tools.gui.remote.method_spec import MethodSpec

from .method_spec_groups.analysis import SPECS as ANALYSIS_SPECS
from .method_spec_groups.arb_waveform import SPECS as ARB_WAVEFORM_SPECS
from .method_spec_groups.connection_device import SPECS as CONNECTION_DEVICE_SPECS
from .method_spec_groups.context import SPECS as CONTEXT_SPECS
from .method_spec_groups.editor import SPECS as EDITOR_SPECS
from .method_spec_groups.notify import SPECS as NOTIFY_SPECS
from .method_spec_groups.operation import SPECS as OPERATION_SPECS
from .method_spec_groups.predictor import SPECS as PREDICTOR_SPECS
from .method_spec_groups.run_save import SPECS as RUN_SAVE_SPECS
from .method_spec_groups.state_project import SPECS as STATE_PROJECT_SPECS
from .method_spec_groups.tab import SPECS as TAB_SPECS
from .method_spec_groups.view import SPECS as VIEW_SPECS
from .method_spec_groups.writeback import SPECS as WRITEBACK_SPECS


def _merge_spec_groups(*groups: Mapping[str, MethodSpec]) -> dict[str, MethodSpec]:
    merged: dict[str, MethodSpec] = {}
    duplicates: set[str] = set()
    for group in groups:
        overlap = set(merged).intersection(group)
        duplicates.update(overlap)
        merged.update(group)
    if duplicates:
        methods = ", ".join(sorted(duplicates))
        raise RuntimeError(f"duplicate remote method specs: {methods}")
    return merged


METHOD_SPECS: dict[str, MethodSpec] = _merge_spec_groups(
    TAB_SPECS,
    RUN_SAVE_SPECS,
    ANALYSIS_SPECS,
    CONTEXT_SPECS,
    STATE_PROJECT_SPECS,
    ARB_WAVEFORM_SPECS,
    CONNECTION_DEVICE_SPECS,
    OPERATION_SPECS,
    VIEW_SPECS,
    PREDICTOR_SPECS,
    WRITEBACK_SPECS,
    EDITOR_SPECS,
    NOTIFY_SPECS,
)
