from threading import Lock
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from IPython.display import display

from ..base import AbsLivePlotter
from ..segments import AbsSegment


class JupyterLivePlotter(AbsLivePlotter):
    """live plotters in Jupyter notebooks."""

    def __init__(
        self, segments: List[AbsSegment], figsize: Optional[Tuple[int, int]] = None
    ) -> None:
        if len(segments) == 0:
            raise ValueError("At least one segment is required.")

        if figsize is None:
            figsize = (5 * len(segments), 4)

        fig, axs = plt.subplots(1, len(segments), figsize=figsize)

        if isinstance(axs, plt.Axes):
            axs = [axs]

        self.fig = fig
        self.axs = axs
        self.segments = segments
        self.update_lock = Lock()

        self.dh = display(self.fig, display_id=True)

    def clear(self) -> None:
        with self.update_lock:
            for ax, segment in zip(self.axs, self.segments):
                segment.clear(ax)
            self._refresh_unchecked()

    def update(self, *args, **kwargs) -> None:
        raise NotImplementedError("Subclass must implement this method.")

    def _refresh_unchecked(self) -> None:
        self.dh.update(self.fig)

    def refresh(self) -> None:
        with self.update_lock:
            self._refresh_unchecked()

    def __enter__(self) -> "JupyterLivePlotter":
        for ax, segment in zip(self.axs, self.segments):
            segment.init_ax(ax)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        plt.close(self.fig)
        pass
