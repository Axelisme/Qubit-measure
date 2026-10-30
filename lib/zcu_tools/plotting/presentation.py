"""Matplotlib manager that routes Figure.show through an explicit owner."""

from collections.abc import Callable

from matplotlib.backend_bases import FigureCanvasBase, FigureManagerBase


class FigureShowManager(FigureManagerBase):
    """Route native Figure.show to a synchronous presentation callback.

    canvas is the figure's current canvas. present takes no arguments and owns
    lifetime checks, thread dispatch and presentation errors. This manager does
    not register with pyplot or choose a backend. Reattach it when a host swaps
    canvases so Figure.show keeps the same presentation contract.
    """

    def __init__(self, canvas: FigureCanvasBase, present: Callable[[], None]) -> None:
        self._present: Callable[[], None] | None = present
        super().__init__(canvas, num=0)

    def show(self) -> None:
        """Present through the bound owner, propagating lifetime and host errors."""
        if self._present is None:
            raise RuntimeError("Plot operation has finished")
        self._present()

    def retire(self) -> None:
        """Drop the operation callback; subsequent show raises RuntimeError."""
        self._present = None

    def attach(self, canvas: FigureCanvasBase) -> None:
        """Bind this manager to a replacement canvas for the same native figure."""
        self.canvas = canvas
        canvas.manager = self
