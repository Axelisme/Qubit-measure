from abc import ABC, abstractmethod

from matplotlib.axes import Axes


class AbsSegment(ABC):
    @abstractmethod
    def init_ax(self, ax: Axes) -> None:
        """Initialize the segment with a matplotlib Axes object."""
        pass

    @abstractmethod
    def update(self, ax: Axes, *args, **kwargs) -> None:
        """Update the segment with new data."""
        pass

    @abstractmethod
    def clear(self, ax: Axes) -> None:
        """Clear the segment from the Axes."""
        pass
