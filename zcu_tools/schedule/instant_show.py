from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from ipywidgets import widgets

from zcu_tools.analysis.tools import minus_mean, rescale


class InstantShow:
    def __init__(
        self,
        *ticks,
        x_label: str,
        y_label: str,
        title: Optional[str] = None,
        prog=None,
        **kwargs,
    ):
        if len(ticks) > 2 or len(ticks) == 0:
            raise ValueError("Invalid number of ticks")

        self.is_1d = len(ticks) == 1
        self.prog = prog

        self._init_widget()
        self._init_fig(*ticks, x_label=x_label, y_label=y_label, title=title, **kwargs)

        wds = widgets.HBox(
            [
                self.fig.canvas,
                widgets.VBox(
                    [
                        self.mode_tb,
                        self.axis_tb,
                        widgets.HBox(
                            [
                                self.minusMean_cb,
                                self.rescale_cb,
                            ]
                        ),
                        self.earlyStop_bt,
                    ]
                ),
            ]
        )

        self.dh = display(*wds, display_id=True)

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

    def _init_widget(self):
        self.mode_tb = widgets.Dropdown(
            options=["Real", "Imag", "Magnitude", "Phase"],
            value="Magnitude",
            description="showing mode",
            layout=widgets.Layout(display="block"),
        )
        self.axis_tb = widgets.Dropdown(
            options=["None", "x", "y"],
            value="None",
            description="operate axis",
            layout=widgets.Layout(display="none" if self.is_1d else "block"),
        )
        self.minusMean_cb = widgets.Checkbox(
            value=False,
            description="minus mean",
            layout=widgets.Layout(display="flex"),
        )
        self.rescale_cb = widgets.Checkbox(
            value=False,
            description="rescale",
            layout=widgets.Layout(display="flex"),
        )
        self.earlyStop_bt = widgets.Button(
            description="Early Stop",
            button_style="danger",
            layout=widgets.Layout(display="none" if self.prog is None else "block"),
        )

        if self.prog is not None:
            self.earlyStop_bt.on_click(self.prog.set_early_stop, remove=True)

    def _process_signals(self, signals: np.ndarray):
        if self.is_1d:
            axis = None
        else:
            map_table = {"None": None, "x": 1, "y": 0}
            axis = map_table[self.axis_tb.value]

        if self.minusMean_cb.value:
            signals = minus_mean(signals, axis)
        if self.rescale_cb.value:
            signals = rescale(signals, axis)

        return signals

    def _cast2real(self, signals: np.ndarray):
        mode = self.mode_tb.value
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
