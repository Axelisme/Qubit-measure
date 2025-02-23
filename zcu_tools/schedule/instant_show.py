import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display


class InstantShow:
    def __init__(self, *ticks, x_label, y_label, title=None, **kwargs):
        if len(ticks) > 2 or len(ticks) == 0:
            raise ValueError("Invalid number of ticks")

        fig, ax = plt.subplots()
        fig.tight_layout(pad=3)  # type: ignore
        ax.set_xlabel(x_label)  # type: ignore
        ax.set_ylabel(y_label)  # type: ignore
        if title:
            ax.set_title(title)  # type: ignore

        self.fig = fig
        self.ax = ax
        if len(ticks) == 1:
            kwargs.setdefault("linestyle", "-")
            kwargs.setdefault("marker", ".")
            self.contain = ax.plot(ticks[0], np.zeros_like(ticks[0]), **kwargs)[0]  # type: ignore
        elif len(ticks) == 2:
            kwargs.setdefault("origin", "lower")
            kwargs.setdefault("interpolation", "none")
            kwargs.setdefault("aspect", "auto")
            self.contain = ax.imshow(  # type: ignore
                np.zeros((len(ticks[1]), len(ticks[0]))),
                extent=[ticks[0][0], ticks[0][-1], ticks[1][0], ticks[1][-1]],
                **kwargs,
            )
        self.dh = display(fig, display_id=True)

    def update_show(self, data, ticks=None):
        if len(data.shape) == 1:  # 1D
            if ticks is None:
                self.contain.set_xdata(ticks)
            self.contain.set_ydata(data)
            self.ax.relim()  # type: ignore
            if ticks is None:
                self.ax.autoscale(axis="y")  # type: ignore
            else:
                self.ax.autoscale_view()  # type: ignore
        elif len(data.shape) == 2:  # 2D
            if ticks is not None:
                X, Y = ticks
                self.contain.set_extent([X[0], X[-1], Y[0], Y[-1]])
            self.contain.set_data(data.T)
            self.contain.autoscale()

        self.dh.update(self.fig)  # type: ignore

    def close_show(self):
        plt.close(self.fig)
