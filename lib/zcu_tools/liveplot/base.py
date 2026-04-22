from __future__ import annotations

from abc import ABC, abstractmethod


class AbsLivePlot(ABC):
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

    def __enter__(self) -> AbsLivePlot:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass


class DummyPlot(AbsLivePlot):
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
