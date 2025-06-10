from threading import Lock
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from IPython.display import display

from ..segments import AbsSegment


class JupyterLivePlotter:
    """live plotters in Jupyter notebooks."""

    def __init__(
        self,
        segments: List[AbsSegment],
        figsize: Optional[Tuple[int, int]] = None,
        disable: bool = False,
    ) -> None:
        if len(segments) == 0:
            raise ValueError("At least one segment is required.")

        if figsize is None:
            figsize = (6 * len(segments), 5)

        if not disable:
            fig, axs = plt.subplots(1, len(segments), figsize=figsize)

            if isinstance(axs, plt.Axes):
                axs = [axs]

            self.fig = fig
            self.axs = axs
            self.dh = display(self.fig, display_id=True)
        else:
            self.fig = None
            self.axs = [None] * len(segments)
            self.dh = None
        self.segments = segments
        self.update_lock = Lock()
        self.disable = disable

    def clear(self) -> None:
        if self.disable:
            return

        with self.update_lock:
            for ax, segment in zip(self.axs, self.segments):
                segment.clear(ax)
            self._refresh_unchecked()

    def _refresh_unchecked(self) -> None:
        if self.disable:
            return
        self.dh.update(self.fig)

    def refresh(self) -> None:
        if self.disable:
            return
        with self.update_lock:
            self._refresh_unchecked()

    def __enter__(self) -> "JupyterLivePlotter":
        if self.disable:
            return self
        for ax, segment in zip(self.axs, self.segments):
            segment.init_ax(ax)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.disable:
            return
        plt.close(self.fig)
