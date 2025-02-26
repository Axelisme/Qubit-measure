from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from ipywidgets import widgets

from zcu_tools.analysis.tools import minus_mean, rescale


class InstantShow:
    def __init__(
        self, *ticks, x_label: str, y_label: str, title: Optional[str] = None, **kwargs
    ):
        if len(ticks) > 2 or len(ticks) == 0:
            raise ValueError("Invalid number of ticks")

        self.is_1d = len(ticks) == 1

        self._init_widget()
        self._init_fig(*ticks, x_label=x_label, y_label=y_label, title=title, **kwargs)

    def _init_fig(
        self, *ticks, x_label: str, y_label: str, title: Optional[str], **kwargs
    ):
        fig, ax = plt.subplots()
        fig.tight_layout(pad=3)  
        ax.set_xlabel(x_label)  
        ax.set_ylabel(y_label)  
        if title:
            ax.set_title(title)  

        self.fig = fig
        self.ax = ax
        if self.is_1d:
            kwargs.setdefault("linestyle", "-")
            kwargs.setdefault("marker", ".")
            self.contain = ax.plot(ticks[0], np.zeros_like(ticks[0]), **kwargs)[0]  
        else:
            kwargs.setdefault("origin", "lower")
            kwargs.setdefault("interpolation", "none")
            kwargs.setdefault("aspect", "auto")
            xs, ys = ticks  
            self.contain = ax.imshow(  
                np.zeros((len(ys), len(xs))),
                extent=[xs[0], xs[-1], ys[0], ys[-1]],
                **kwargs,
            )

        self.dh = display(fig, display_id=True)

    def _init_widget(self):
        if not self.is_1d:
            self.axis_widget = widgets.Dropdown(
                options=["None", "x", "y"],
                value="None",
                description="operate axis",
            )
        self.minus_mean_widget = widgets.Checkbox(
            value=False,
            description="minus mean",
        )
        self.rescale_widget = widgets.Checkbox(
            value=False,
            description="rescale",
        )
        self.mode_widget = widgets.Dropdown(
            options=["Real", "Imag", "Magnitude", "Phase"],
            value="Magnitude",
            description="showing mode",
        )

        widget_list = [self.minus_mean_widget, self.rescale_widget, self.mode_widget]
        if not self.is_1d:
            widget_list.insert(0, self.axis_widget)
        display(*widget_list)

    def _process_signals(self, signals: np.ndarray):
        if self.is_1d:
            axis = None
        else:
            map_table = {"None": None, "x": 1, "y": 0}
            axis = map_table[self.axis_widget.value]  

        if self.minus_mean_widget.value:
            signals = minus_mean(signals, axis)
        if self.rescale_widget.value:
            signals = rescale(signals, axis)

        return signals

    def _cast2real(self, signals: np.ndarray):
        mode = self.mode_widget.value
        if mode == "Real":
            return np.real(signals)
        elif mode == "Imag":
            return np.imag(signals)
        elif mode == "Magnitude":
            return np.abs(signals)
        elif mode == "Phase":
            return np.unwrap(np.angle(signals))
        else:
            raise ValueError(f"未知的模式: {mode}")

    def update_show(self, signals: np.ndarray, ticks=None):
        if self.is_1d:  # 1D
            self._update_show1d(signals, ticks)

        else:  # 2D
            self._update_show2d(signals, ticks)

        self.dh.update(self.fig)  

    def _update_show1d(self, signals: np.ndarray, ticks):
        if len(signals.shape) != 1:
            raise ValueError("Invalid shape of signals")

        signals = self._process_signals(signals)
        ys = self._cast2real(signals)

        if ticks is None:
            self.contain.set_xdata(ticks)
        self.contain.set_ydata(ys)

        self.ax.relim()  
        if ticks is None:
            self.ax.autoscale(axis="y")  
        else:
            self.ax.autoscale_view()  

    def _update_show2d(self, signals: np.ndarray, ticks):
        if len(signals.shape) != 2:
            raise ValueError("Invalid shape of signals")

        signals = self._process_signals(signals.T)
        zs = self._cast2real(signals)

        if ticks is not None:
            X, Y = ticks
            self.contain.set_extent([X[0], X[-1], Y[0], Y[-1]])

        self.contain.set_data(zs)
        self.contain.autoscale()

    def close_show(self):
        plt.close(self.fig)
