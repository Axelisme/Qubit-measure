from abc import ABC, abstractmethod

import matplotlib.pyplot as plt


class AbsSegment(ABC):
    @abstractmethod
    def init_ax(self, ax: plt.Axes) -> None:
        """Initialize the segment with a matplotlib Axes object."""
        pass

    @abstractmethod
    def update(self, ax: plt.Axes, *args, **kwargs) -> None:
        """Update the segment with new data."""
        pass

    @abstractmethod
    def clear(ax: plt.Axes) -> None:
        """Clear the segment from the Axes."""
        pass
