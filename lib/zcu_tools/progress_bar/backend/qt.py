"""Qt progress bar backend — thin wrapper around ProgressModel.

ProgressModel (in gui/services/device_progress.py) owns the thread-safe
signal bridge and the ProgressStack widget connection.  This module just
wires the public pbar-factory API to it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from zcu_tools.gui.services.device_progress import (
    ProgressBar,
    ProgressEntrySnapshot,
    ProgressFactory,
    ProgressModel,
)

if TYPE_CHECKING:
    from zcu_tools.gui.ui.progress_stack import ProgressStack

# Re-export snapshot type so existing imports of RunProgressSnapshot keep working.
RunProgressSnapshot = ProgressEntrySnapshot


class QtProgressBarFactory(ProgressFactory):
    """ProgressFactory backed by a ProgressModel.

    Pass either an existing ProgressModel (preferred, avoids double-attach)
    or a ProgressStack (creates a new model and attaches it).
    """

    def __init__(self, model_or_stack: "ProgressModel | ProgressStack") -> None:
        if isinstance(model_or_stack, ProgressModel):
            model = model_or_stack
        else:
            model = ProgressModel()
            model.attach_stack(model_or_stack)
        super().__init__(model)

    def get_all_snapshots(self) -> tuple[ProgressEntrySnapshot, ...]:
        return self._model.snapshot()

    def set_progress_callback(  # noqa: ARG002
        self,
        cb: Any,
        interval: int = 10,  # noqa: ARG002
    ) -> None:
        pass  # no longer needed — widget updates directly from model.changed


# Keep old names importable from tests / other code.
QtProgressBar = ProgressBar
_FLOAT_SCALE = 10000  # noqa: F401  re-export for tests
