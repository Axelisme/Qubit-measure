from abc import ABC, abstractmethod
from typing import Dict, Any


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


class NonePlotter(AbsLivePlotter):
    """
    A plotter that does nothing.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def clear(self) -> None:
        pass

    def update(self, *args, refresh: bool = True, **kwargs) -> None:
        pass

    def refresh(self) -> None:
        pass


class MultiLivePlotter(AbsLivePlotter):
    """
    A wrapper for multiple live plotters.

    This class need a dispatch function to dispatch the arguments to the plotters.
    The dispatch function should return a dictionary of arguments for each plotter.
    If the argument is None, the corresponding plotter will not be updated.
    """

    def __init__(
        self,
        plotters: Dict[str, AbsLivePlotter],
    ) -> None:
        self.plotters = plotters

    def clear(self) -> None:
        for plotter in self.plotters.values():
            plotter.clear()

    def update(self, plot_args: Dict[str, Any], refresh: bool = True) -> None:
        for name, args_i in plot_args.items():
            self.plotters[name].update(*args_i, refresh=refresh)

    def refresh(self) -> None:
        for plotter in self.plotters.values():
            plotter.refresh()

    def __enter__(self) -> "MultiLivePlotter":
        for plotter in self.plotters.values():
            plotter.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for plotter in self.plotters.values():
            plotter.__exit__(exc_type, exc_value, traceback)
