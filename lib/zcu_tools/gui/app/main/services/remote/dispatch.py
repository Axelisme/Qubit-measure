"""Method registry composition root for RemoteControlAdapter.

Handler implementations live in ``handlers/``. This module keeps the public
``METHOD_REGISTRY`` import path stable for the service layer and tests.
"""

from __future__ import annotations

from collections.abc import Mapping

from zcu_tools.gui.remote.method_spec import BoundMethod, build_method_registry

from .handlers._common import Handler
from .handlers.analysis import HANDLERS as ANALYSIS_HANDLERS
from .handlers.arb_waveform import HANDLERS as ARB_WAVEFORM_HANDLERS
from .handlers.connection_device import HANDLERS as CONNECTION_DEVICE_HANDLERS
from .handlers.context import HANDLERS as CONTEXT_HANDLERS
from .handlers.editor import HANDLERS as EDITOR_HANDLERS
from .handlers.notify import HANDLERS as NOTIFY_HANDLERS
from .handlers.operation import HANDLERS as OPERATION_HANDLERS
from .handlers.predictor import HANDLERS as PREDICTOR_HANDLERS
from .handlers.run_save import HANDLERS as RUN_SAVE_HANDLERS
from .handlers.state_project import HANDLERS as STATE_PROJECT_HANDLERS
from .handlers.tab import HANDLERS as TAB_HANDLERS
from .handlers.view import HANDLERS as VIEW_HANDLERS
from .handlers.writeback import HANDLERS as WRITEBACK_HANDLERS
from .method_specs import METHOD_SPECS


def _merge_handler_groups(*groups: Mapping[str, Handler]) -> dict[str, Handler]:
    merged: dict[str, Handler] = {}
    duplicates: set[str] = set()
    for group in groups:
        overlap = set(merged).intersection(group)
        duplicates.update(overlap)
        merged.update(group)
    if duplicates:
        methods = ", ".join(sorted(duplicates))
        raise RuntimeError(f"duplicate remote handler methods: {methods}")
    return merged


_HANDLERS: dict[str, Handler] = _merge_handler_groups(
    TAB_HANDLERS,
    RUN_SAVE_HANDLERS,
    ANALYSIS_HANDLERS,
    CONTEXT_HANDLERS,
    STATE_PROJECT_HANDLERS,
    ARB_WAVEFORM_HANDLERS,
    CONNECTION_DEVICE_HANDLERS,
    OPERATION_HANDLERS,
    VIEW_HANDLERS,
    PREDICTOR_HANDLERS,
    WRITEBACK_HANDLERS,
    EDITOR_HANDLERS,
    NOTIFY_HANDLERS,
)

METHOD_REGISTRY: dict[str, BoundMethod] = build_method_registry(_HANDLERS, METHOD_SPECS)
