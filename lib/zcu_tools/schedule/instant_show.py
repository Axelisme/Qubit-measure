# type: ignore

from typing import Optional
from threading import Lock

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display


class BaseInstantShow:
    def __init__(self, x_label: str, y_label: str, title: Optional[str] = None):
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
        if hasattr(self, "fig"):
            plt.close(self.fig)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close_show()


class InstantShow1D(BaseInstantShow):
    def __init__(
        self, xs, x_label: str, y_label: str, title: Optional[str] = None, **kwargs
    ):
        super().__init__(x_label, y_label, title)

        kwargs.setdefault("linestyle", "-")
        kwargs.setdefault("marker", ".")
        self.contain = self.ax.plot(xs, np.zeros_like(xs), **kwargs)[0]

        self.dh = display(self.fig, display_id=True)

    def update_show(self, signals_real: np.ndarray, ticks=None, title=None):
        with self.update_lock:
            if len(signals_real.shape) != 1:
                raise ValueError(
                    f"Invalid shape of signals: {signals_real.shape}, expect 1D"
                )

            if ticks is not None:
                self.contain.set_xdata(ticks)
            self.contain.set_ydata(signals_real)

            if title:
                self.ax.set_title(title)

            self.ax.relim(visible_only=True)
            if ticks is None:
                self.ax.autoscale(axis="y")
            else:
                self.ax.autoscale_view()

            self.dh.update(self.fig)


class InstantShow2D(BaseInstantShow):
    def __init__(
        self, xs, ys, x_label: str, y_label: str, title: Optional[str] = None, **kwargs
    ):
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

    def update_show(self, signals_real: np.ndarray, ticks=None, title=None):
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
    def __init__(
        self, x_label: str, y_label: str, title: Optional[str] = None, **kwargs
    ):
        super().__init__(x_label, y_label, title)

        self.contain = self.ax.scatter([], [], **kwargs)

        self.xs = []
        self.ys = []
        self.cs = []

        self.dh = display(self.fig, display_id=True)

    def append_spot(self, x, y, color, title=None):
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
