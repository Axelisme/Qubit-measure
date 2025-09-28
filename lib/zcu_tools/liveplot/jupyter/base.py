from itertools import chain
from threading import Lock
from typing import List, Optional, TypeVar

import matplotlib.pyplot as plt
from IPython.display import display

from ..segments import AbsSegment

# Generic type variable used to correctly type the context-manager methods so that
# subclasses don't need to override ``__enter__`` just to narrow the return type.
T_JupyterPlotMixin = TypeVar("T_JupyterPlotMixin", bound="JupyterPlotMixin")


class JupyterPlotMixin:
    """live plotters in Jupyter notebooks."""

    def __init__(
        self,
        segments: List[List[AbsSegment]],
        provide_axs: Optional[List[List[plt.Axes]]] = None,
        disable: bool = False,
    ) -> None:
        if len(chain.from_iterable(segments)) == 0:
            raise ValueError("At least one segment is required.")
        n_row = len(segments)
        n_col = len(segments[0])

        # validate check
        for s_row in segments:
            if len(s_row) != n_col:
                raise ValueError(
                    "Number of segments in each row must match number of columns."
                )

        self.fig = None
        self.axs = [[None for _ in range(n_col)] for _ in range(n_row)]
        self.dh = None
        self.host_by_self = False
        if provide_axs is not None:
            # validate check
            valid = len(provide_axs) == n_row
            for a_row in provide_axs:
                if len(a_row) != n_col:
                    valid = False
            if not valid:
                raise ValueError(
                    "The shape of provided axes must match the shape of segments."
                )

            self.axs = provide_axs

        # if not provided axes, create figure and display handle
        if provide_axs is None and not disable:
            self.fig = plt.figure(figsize=(6 * n_col, 5 * n_row))
            assert isinstance(self.fig, plt.FigureBase)

            self.axs = self.fig.subplots(n_row, n_col, squeeze=False)
            self.dh = display(self.fig, display_id=True)

            self.host_by_self = True  # host figure by self

        self.segments = segments
        self.update_lock = Lock()
        self.disable = disable

    def clear(self) -> None:
        if self.disable:
            return

        with self.update_lock:
            for ax_row, seg_row in zip(self.axs, self.segments):
                for ax, segment in zip(ax_row, seg_row):
                    segment.clear(ax)
            self._refresh_while_lock()

    def _refresh_while_lock(self) -> None:
        assert self.update_lock.locked()

        if self.disable:
            return

        if self.host_by_self:
            assert self.dh is not None
            assert self.fig is not None
            self.dh.update(self.fig)

    def refresh(self) -> None:
        if self.disable:
            return

        with self.update_lock:
            self._refresh_while_lock()

    def __enter__(self: T_JupyterPlotMixin) -> T_JupyterPlotMixin:
        if self.disable:
            return self

        for ax_row, seg_row in zip(self.axs, self.segments):
            for ax, segment in zip(ax_row, seg_row):
                segment.init_ax(ax)

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.disable:
            return

        if self.host_by_self:
            assert self.fig is not None
            plt.close(self.fig)
