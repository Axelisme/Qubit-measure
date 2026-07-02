"""Single registration source for measure-gui remote methods."""

from __future__ import annotations

from ._registry import RemoteMethodEntry
from .analysis import METHODS as ANALYSIS_METHODS
from .arb_waveform import METHODS as ARB_WAVEFORM_METHODS
from .connection_device import METHODS as CONNECTION_DEVICE_METHODS
from .context import METHODS as CONTEXT_METHODS
from .editor import METHODS as EDITOR_METHODS
from .notify import METHODS as NOTIFY_METHODS
from .operation import METHODS as OPERATION_METHODS
from .predictor import METHODS as PREDICTOR_METHODS
from .run_save import METHODS as RUN_SAVE_METHODS
from .state_project import METHODS as STATE_PROJECT_METHODS
from .tab import METHODS as TAB_METHODS
from .view import METHODS as VIEW_METHODS
from .writeback import METHODS as WRITEBACK_METHODS

METHOD_ENTRIES: tuple[RemoteMethodEntry, ...] = (
    *TAB_METHODS,
    *RUN_SAVE_METHODS,
    *ANALYSIS_METHODS,
    *CONTEXT_METHODS,
    *STATE_PROJECT_METHODS,
    *ARB_WAVEFORM_METHODS,
    *CONNECTION_DEVICE_METHODS,
    *OPERATION_METHODS,
    *VIEW_METHODS,
    *PREDICTOR_METHODS,
    *WRITEBACK_METHODS,
    *EDITOR_METHODS,
    *NOTIFY_METHODS,
)
