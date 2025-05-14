import numpy as np
from myqick.asm_v2 import QickParam, QickProgramV2
from zcu_tools.program.base import MyProgram

from .pulse import Pulse, pulses_to_signal


class SimulateV2(MyProgram, QickProgramV2):
    """
    Record the pulse sequence in a list of Pulse objects, So we can plot them later.
    It is performed by overriding the delay and pulse methods.
    It isn't very accurate, but it is enough for most cases.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sim_cur_t = 0.0
        self.pulse_end_t = 0.0
        self.pulse_list = []

    def delay(self, t, tag=None):
        super().delay(t, tag=tag)

        self.sim_cur_t = t

    def delay_auto(self, t=0, gens=True, ros=True, tag=None):
        super().delay_auto(t, gens=gens, ros=ros, tag=tag)

        self.sim_cur_t += self.pulse_end_t + t

    def pulse(self, ch, name, t=0, tag=None):
        super().pulse(ch, name, t=t, tag=tag)

        start_t = self.sim_cur_t + t
        pulse_cfg = self.pulse_map[name]
        self.pulse_list.append(Pulse(start_t, pulse_cfg))

        if start_t + pulse_cfg["length"] > self.pulse_end_t:
            self.pulse_end_t = start_t + pulse_cfg["length"]

    def visualize(self):
        total_length = self.sim_cur_t
        if isinstance(total_length, QickParam):
            total_length = total_length.maxval()

        NUM_SAMPLE = 1001

        times = np.linspace(0.0, total_length, NUM_SAMPLE)
        signal_dict = pulses_to_signal(self.pulse_list, times)

        # remove unused dimensions
        visualize_keywords = ["length", "sigma", "alpha", "gain"]
        use_dims = [
            i
            for i, name in enumerate(self.loop_dict.keys())
            if any(kw in name for kw in visualize_keywords)
        ]

        import ipywidgets as widgets
        import matplotlib.pyplot as plt
        from IPython.display import display

        signal_shape = next(iter(signal_dict.values())).shape

        # 根據use_indices判斷哪些維度需要滑桿
        slider_names = [self.loop_dict.keys()[i] for i in use_dims]
        slider_counts = [self.loop_dict[name] for name in slider_names]

        def plot_func(*slider_vals):
            # 根據滑桿值組合index
            indices = [0] * len(signal_shape)
            for i, val in zip(use_dims, slider_vals):
                indices[i] = val
            indices = tuple(indices)

            plt.figure(figsize=(10, 4))
            for ch, sig in signal_dict.items():
                # sig.shape = (..., NUM_SAMPLE)
                plt.plot(times, np.real(sig[indices]), label=f"ch{ch} (real)")
                plt.plot(times, np.imag(sig[indices]), "--", label=f"ch{ch} (imag)")
            plt.xlabel("Time (us)")
            plt.ylabel("Amplitude")
            plt.title("Pulse Sequence Simulation")
            plt.legend()
            plt.grid(True)
            plt.show()

        if len(slider_names) > 0:
            sliders = [
                widgets.IntSlider(min=0, max=c - 1, step=1, description=n)
                for n, c in zip(slider_names, slider_counts)
            ]
            ui = widgets.HBox(sliders)
            out = widgets.interactive_output(
                plot_func, {n: s for n, s in zip(slider_names, sliders)}
            )
            display(ui, out)
        else:
            plot_func()
