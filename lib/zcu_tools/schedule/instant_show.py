# type: ignore

from threading import Lock
from typing import Optional

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
        if np.all(np.isnan(errs)):
            return errs

        s_len = max(len(errs) // 20, 1)
        _errs = np.full_like(errs, np.nan)
        for i in range(len(errs)):
            start = max(0, i - s_len)
            end = min(len(errs), i + s_len)
            if start == end:
                _errs[i] = np.nan
            else:
                _errs[i] = np.nanmedian(errs[start:end])
        _errs[np.isnan(errs)] = np.nan
        return _errs

    def update_show(
        self,
        signals_real: np.ndarray,
        *,
        errs: np.ndarray = None,
        ticks=None,
        title=None,
    ):
        if errs is None:
            errs = np.full_like(signals_real, np.nan)

        if ticks is not None:
            self.xs = ticks

        sorted_idxs = np.argsort(self.xs)
        sorted_xs = self.xs[sorted_idxs]
        signals_real = signals_real[sorted_idxs]
        errs = errs[sorted_idxs]

        # smooth error bars
        errs_up = self._smooth_errs(signals_real + 2 * errs)
        errs_dn = self._smooth_errs(signals_real - 2 * errs)

        # make errs_up and errs_dn inclusive signals_real
        # move_up = np.clip(signals_real - errs_up, 0, None)
        # move_dn = np.clip(errs_dn - signals_real, 0, None)
        # errs_up = errs_up + move_up - move_dn
        # errs_dn = errs_dn + move_up - move_dn

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

    def update_show(
        self, signals_real: np.ndarray, *, errs=None, ticks=None, title=None
    ):
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

    def append_spot(self, x, y, color, *, title=None):
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
