# type: ignore
"""
Real-time data visualization tools for Jupyter notebooks.

This module provides classes for instantly displaying and updating plots in Jupyter notebooks,
including 1D line plots, 2D heatmaps, and scatter plots.
"""

from threading import Lock
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display


class BaseInstantShow:
    """
    Base class for instant visualization in Jupyter notebooks.

    This class provides common functionality for creating and managing
    interactive plots that can be updated in real-time.

    Attributes:
        fig: The matplotlib figure object
        ax: The matplotlib axes object
        update_lock: A threading lock to prevent concurrent updates to the plot
    """

    def __init__(self, x_label: str, y_label: str, title: Optional[str] = None):
        """
        Initialize a base instant show plot.

        Args:
            x_label: Label for the x-axis
            y_label: Label for the y-axis
            title: Optional title for the plot
        """
        fig, ax = plt.subplots()
        fig.tight_layout(pad=3)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if title:
            ax.set_title(title)

        self.fig = fig
        self.ax = ax
        self.update_lock = Lock()

    def close_show(self):
        """Close the matplotlib figure to free resources."""
        if hasattr(self, "fig"):
            plt.close(self.fig)

    def __enter__(self):
        """Context manager entry point to support 'with' statement."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit point that ensures resources are released."""
        self.close_show()


class InstantShow1D(BaseInstantShow):
    """
    Class for real-time visualization of 1D data with error bars.

    This class creates an interactive line plot that can be updated
    with new data points and error ranges.

    Attributes:
        xs: The x-axis values
        contain: The main plot line
        err_up: The upper error bound line
        err_dn: The lower error bound line
        dh: The IPython display handle
    """

    def __init__(
        self, xs, x_label: str, y_label: str, title: Optional[str] = None, **kwargs
    ):
        """
        Initialize a 1D instant show plot.

        Args:
            xs: Array of x-axis values
            x_label: Label for the x-axis
            y_label: Label for the y-axis
            title: Optional title for the plot
            **kwargs: Additional keyword arguments to pass to matplotlib's plot function
        """
        super().__init__(x_label, y_label, title)

        self.xs = xs
        sorted_xs = xs[np.argsort(xs)]

        err_kwargs = {"linestyle": "--", "color": "lightgray"}
        self.err_up = self.ax.plot(sorted_xs, np.zeros_like(xs), **err_kwargs)[0]
        self.err_dn = self.ax.plot(sorted_xs, np.zeros_like(xs), **err_kwargs)[0]

        kwargs.setdefault("linestyle", "-")
        kwargs.setdefault("marker", ".")
        self.contain = self.ax.plot(sorted_xs, np.zeros_like(xs), **kwargs)[0]

        self.dh = display(self.fig, display_id=True)

    def _smooth_errs(self, errs):
        """
        Smooth error values using median filtering.

        Args:
            errs: Array of error values to smooth

        Returns:
            Smoothed error array with the same shape as input
        """
        if np.all(np.isnan(errs)):
            return errs

        s_len = max(len(errs) // 20, 1)
        _errs = np.full_like(errs, np.nan)
        for i in range(len(errs)):
            start = max(0, i - s_len)
            end = min(len(errs), i + s_len)
            if start == end or np.all(np.isnan(errs[start:end])):
                _errs[i] = np.nan
            else:
                _errs[i] = np.nanmedian(errs[start:end])
        _errs[np.isnan(errs)] = np.nan
        return _errs

    def update_show(
        self,
        signals_real: np.ndarray,
        *,
        errs: Optional[np.ndarray] = None,
        ticks: Optional[np.ndarray] = None,
        title: Optional[str] = None,
    ):
        """
        Update the plot with new data and error values.

        Args:
            signals_real: Array of y-values to display
            errs: Optional array of error values corresponding to signals_real
            ticks: Optional new x-axis values to replace the existing ones
            title: Optional new title for the plot
        """
        if errs is None:
            errs = np.full_like(signals_real, np.nan)

        if ticks is not None:
            self.xs = ticks

        sorted_idxs = np.argsort(self.xs)
        sorted_xs = self.xs[sorted_idxs]
        signals_real = signals_real[sorted_idxs]
        errs = errs[sorted_idxs]

        # smooth error bars
        errs_up = signals_real + 2 * errs
        errs_dn = signals_real - 2 * errs
        # errs_up = self._smooth_errs(signals_real + 2 * errs)
        # errs_dn = self._smooth_errs(signals_real - 2 * errs)

        with self.update_lock:
            self.contain.set_data(sorted_xs, signals_real)
            self.err_up.set_data(sorted_xs, errs_up)
            self.err_dn.set_data(sorted_xs, errs_dn)

            if title:
                self.ax.set_title(title)

            self.ax.relim(visible_only=True)
            self.ax.autoscale_view()

            self.dh.update(self.fig)


class InstantShow2D(BaseInstantShow):
    """
    Class for real-time visualization of 2D data as heatmaps.

    This class creates an interactive 2D heatmap that can be updated
    with new data matrices.

    Attributes:
        contain: The imshow plot object
        dh: The IPython display handle
    """

    def __init__(
        self, xs, ys, x_label: str, y_label: str, title: Optional[str] = None, **kwargs
    ):
        """
        Initialize a 2D instant show plot.

        Args:
            xs: Array of x-axis values
            ys: Array of y-axis values
            x_label: Label for the x-axis
            y_label: Label for the y-axis
            title: Optional title for the plot
            **kwargs: Additional keyword arguments to pass to matplotlib's imshow function
        """
        super().__init__(x_label, y_label, title)

        kwargs.setdefault("origin", "lower")
        kwargs.setdefault("interpolation", "none")
        kwargs.setdefault("aspect", "auto")

        self.contain = self.ax.imshow(
            np.zeros((len(ys), len(xs))),
            extent=[xs[0], xs[-1], ys[0], ys[-1]],
            **kwargs,
        )

        self.dh = display(self.fig, display_id=True)

    def update_show(
        self,
        signals_real: np.ndarray,
        *,
        errs: Optional[np.ndarray] = None,
        ticks: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        title: Optional[str] = None,
    ):
        """
        Update the 2D heatmap with new data.

        Args:
            signals_real: 2D array of values to display as a heatmap
            errs: Optional error values (not used in 2D plots)
            ticks: Optional tuple of (X, Y) arrays to update axis extents
            title: Optional new title for the plot

        Raises:
            ValueError: If signals_real is not a 2D array
        """
        with self.update_lock:
            if len(signals_real.shape) != 2:
                raise ValueError(
                    f"Invalid shape of signals: {signals_real.shape}, expect 2D"
                )

            if ticks is not None:
                X, Y = ticks
                self.contain.set_extent([X[0], X[-1], Y[0], Y[-1]])

            if title:
                self.ax.set_title(title)

            self.contain.set_data(signals_real.T)
            self.contain.autoscale()

            self.dh.update(self.fig)


class InstantShowScatter(BaseInstantShow):
    """
    Class for real-time visualization of scatter plots.

    This class creates an interactive scatter plot that can be dynamically
    extended with new points.

    Attributes:
        contain: The scatter plot object
        xs: List of x-coordinates
        ys: List of y-coordinates
        cs: List of color values
        dh: The IPython display handle
    """

    def __init__(
        self, x_label: str, y_label: str, title: Optional[str] = None, **kwargs
    ):
        """
        Initialize a scatter instant show plot.

        Args:
            x_label: Label for the x-axis
            y_label: Label for the y-axis
            title: Optional title for the plot
            **kwargs: Additional keyword arguments to pass to matplotlib's scatter function
        """
        super().__init__(x_label, y_label, title)

        self.contain = self.ax.scatter([], [], **kwargs)

        self.xs = []
        self.ys = []
        self.cs = []

        self.dh = display(self.fig, display_id=True)

    def append_spot(
        self, x: float, y: float, color: float, *, title: Optional[str] = None
    ):
        """
        Add a new point to the scatter plot.

        Args:
            x: x-coordinate of the new point
            y: y-coordinate of the new point
            color: Color value for the new point (used for colormap)
            title: Optional new title for the plot
        """
        with self.update_lock:
            self.xs.append(x)
            self.ys.append(y)
            self.cs.append(color)

            self.contain.set_offsets(np.column_stack((self.xs, self.ys)))
            self.contain.set_array(np.array(self.cs))
            self.contain.set_clim(min(self.cs), max(self.cs))

            # print(f"x: {x:.1e}, y: {y:.1e}, color: {color:.1e}", end="\r")

            if title:
                self.ax.set_title(title)

            x_min, x_max = min(self.xs), max(self.xs)
            y_min, y_max = min(self.ys), max(self.ys)
            self.ax.set_xlim(min(x_min, x_max - 1e-6), max(x_max, x_min + 1e-6))
            self.ax.set_ylim(min(y_min, y_max - 1e-6), max(y_max, y_min + 1e-6))

            self.dh.update(self.fig)
