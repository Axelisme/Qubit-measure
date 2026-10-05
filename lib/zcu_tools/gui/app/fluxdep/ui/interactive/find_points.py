"""Disposable Qt brush controls and generation-checked TwoTone projections."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from matplotlib.backend_bases import MouseEvent
from numpy.typing import NDArray

from zcu_tools.analysis.fluxdep.twotone import TwoTonePickView
from zcu_tools.gui.app.fluxdep.interactive import TwoTonePickContext

from .base import InteractiveMplWidget

TwoTonePreviewSubmitter = Callable[
    [
        Callable[[], TwoTonePickView],
        Callable[[TwoTonePickView], None],
        Callable[[Exception], None],
    ],
    None,
]


class FindPointsWidget(InteractiveMplWidget):
    """Controls and pointer preview of one app-owned TwoTone selection.

    Session is the sole authoritative state. Worker results are only derived
    views. Finish emits a request to the app owner, which recomputes exact state.
    """

    def __init__(
        self,
        context: TwoTonePickContext,
        *,
        submit_preview: TwoTonePreviewSubmitter | None = None,
    ) -> None:
        """Attach open context and start an 80ms-debounced derived preview.

        None creates an owned BackgroundRunner. An injected submit_preview must
        deliver on owner and its caller owns external worker cleanup. Controls
        use shared Actions/Undo. Failed Actions reproject committed controls
        before showing their error in the Status label; never coerce conflicting
        detector settings. Closed context raises FailedPreconditionError.
        No spectrum ownership or domain cancellation is transferred.
        """
        raise NotImplementedError

    def update_points(self) -> None:
        """Invalidate older views and schedule latest detached state off-main.

        Pending scatter/cache is cleared. Latest failure remains visible and
        preview_view returns None. Old/closed/detached completions are discarded.
        This presentation update never commits or changes Undo.
        """
        raise NotImplementedError

    def preview_view(self) -> TwoTonePickView | None:
        """Return detached latest settled view or None while pending/closed/detached.

        Never recompute inline or expose a stale generation or mutable cache.
        """
        raise NotImplementedError

    def on_press(self, event: MouseEvent) -> None:
        """Start a left-button pointer preview without committing selection."""
        raise NotImplementedError

    def on_move(self, event: MouseEvent) -> None:
        """Append finite in-axis vertices while a gesture is active; no commit."""
        raise NotImplementedError

    def on_release(self, event: MouseEvent) -> None:
        """Commit one complete gesture on latest Session, then clear preview.

        Out-of-axis release uses the last valid vertex. Width/mode are captured
        on press; invalid/closed input is shown without committing. Non-left or
        inactive gestures do not submit.
        """
        raise NotImplementedError

    def get_result(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Synchronously derive sorted native device/GHz points from exact state.

        Never read preview cache or controls; propagate numerical failures.
        """
        raise NotImplementedError

    def quiesce(self) -> None:
        """Idempotently detach/invalidate, stop debounce and join owned runner.

        Unsubscribe and discard late completions without closing Session.
        External injected submitter cleanup remains its caller's responsibility.
        """
        raise NotImplementedError

    def teardown(self) -> None:
        """Idempotently quiesce presentation without cancelling domain input."""
        self.quiesce()
