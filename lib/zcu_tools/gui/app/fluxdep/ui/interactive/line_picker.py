"""Session-backed fluxdep line picking, with disposable pointer preview."""

from __future__ import annotations

from matplotlib.backend_bases import MouseEvent

from zcu_tools.analysis.fluxdep import find_best_mirror_position, fold_initial_lines
from zcu_tools.gui.app.fluxdep.interactive import LinePickContext
from zcu_tools.gui.app.fluxdep.ui.interactive.base import InteractiveMplWidget

__all__ = ["LinePickerWidget", "find_best_mirror_position", "fold_initial_lines"]


class LinePickerWidget(InteractiveMplWidget):
    """Present one owner's live context, without owning committed line state.

    Controls use context.plugin actions and context.session undo. The finished
    signal requests owner Finish; get_result is only a committed-state read.
    Invalid moves restore the snapshot and display their error, not a commit.
    """

    def __init__(self, context: LinePickContext) -> None:
        """Attach to an open context and capture its immutable rendering inputs.

        Closing the context rejects input. Teardown detaches only presentation;
        the owner may mount another view of the same session.
        """
        raise NotImplementedError

    def on_press(self, event: MouseEvent) -> None:
        """Select a line on the main spectrum axes; do not commit."""
        raise NotImplementedError

    def on_move(self, event: MouseEvent) -> None:
        """Preview a valid device position while a line is selected."""
        raise NotImplementedError

    def on_release(self, event: MouseEvent) -> None:
        """Commit a valid selected-line placement through the shared move action."""
        raise NotImplementedError

    def get_result(self) -> tuple[float, float]:
        """Read committed half/integer device positions, not pointer preview."""
        raise NotImplementedError

    def cancel_preview(self) -> None:
        """Restore the latest snapshot on focus loss, hide or Escape."""
        raise NotImplementedError

    def teardown(self) -> None:
        """Idempotently detach subscriptions and controls, leaving context open."""
        raise NotImplementedError
