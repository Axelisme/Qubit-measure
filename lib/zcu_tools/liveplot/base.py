from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple


class AbsLivePlotter(ABC):
    """
    Base class for instant visualization.

    This class provides common functionality for creating and managing
    interactive plots that can be updated in real-time.
    """

    @abstractmethod
    def clear(self) -> None:
        """Clear the plot."""
        pass

    @abstractmethod
    def update(self, *args, refresh: bool = True, **kwargs) -> None:
        """Update the plot with new data."""
        pass

    @abstractmethod
    def refresh(self) -> None:
        """Refresh the plot to reflect the latest data."""
        pass

    def __enter__(self) -> "AbsLivePlotter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass


class MultiLivePlotter(AbsLivePlotter):
    """
    A wrapper for multiple live plotters.

    This class need a dispatch function to dispatch the arguments to the plotters.
    The dispatch function should return a list of arguments for each plotter.
    If the argument is None, the corresponding plotter will not be updated.
    """

    def __init__(
        self,
        plotters: List[AbsLivePlotter],
        dispatch_fn: Callable[..., List[Optional[Tuple]]],
    ) -> None:
        self.plotters = plotters
        self.dispatch_fn = dispatch_fn

    def clear(self) -> None:
        for plotter in self.plotters:
            plotter.clear()

    def update(self, *args, refresh: bool = True, **kwargs) -> None:
        plot_args = self.dispatch_fn(*args, **kwargs)

        if len(plot_args) != len(self.plotters):
            raise ValueError("Number of plotters and plot arguments must match.")

        for plotter_i, args_i in zip(self.plotters, plot_args):
            if args_i is None:
                continue
            plotter_i.update(*args_i, refresh=refresh)

    def refresh(self) -> None:
        for plotter in self.plotters:
            plotter.refresh()

    def __enter__(self) -> "MultiLivePlotter":
        for plotter in self.plotters:
            plotter.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for plotter in self.plotters:
            plotter.__exit__(exc_type, exc_value, traceback)
