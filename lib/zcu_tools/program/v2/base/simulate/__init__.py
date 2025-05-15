import numpy as np
from myqick.asm_v2 import QickParam, QickProgramV2
from zcu_tools.program.base import MyProgram

from .pulse import Pulse, pulses_to_signal
from .waveform import format_param


class SimulateV2(MyProgram, QickProgramV2):
    """
    Record the pulse sequence in a list of Pulse objects, So we can plot them later.
    It is performed by overriding the delay and pulse methods.
    It isn't very accurate, but it is enough for most cases.
    """

    def __init__(self, *args, **kwargs):
        self.sim_ref_t = 0.0

        self.sim_last_gen_end_t = None
        self.sim_last_ro_end_t = None

        self.pulse_list = []
        self.sim_ro_length = None

        super().__init__(*args, **kwargs)

    def delay(self, t, tag=None):
        super().delay(t, tag=tag)

        self.sim_ref_t = max(self.sim_ref_t, t)

    def delay_auto(self, t=0, gens=True, ros=True, tag=None):
        super().delay_auto(t, gens=gens, ros=ros, tag=tag)

        last_end = self.sim_ref_t
        if gens and self.sim_last_gen_end_t is not None:
            last_end = max(last_end, self.sim_last_gen_end_t)
        if ros and self.sim_last_ro_end_t is not None:
            last_end = max(last_end, self.sim_last_ro_end_t)
        self.sim_ref_t = max(self.sim_ref_t, last_end + t)

    def pulse(self, ch, name, t=0, tag=None):
        super().pulse(ch, name, t=t, tag=tag)

        start_t = self.sim_ref_t
        if t == "auto":
            if self.sim_last_gen_end_t is not None:
                start_t = self.sim_last_gen_end_t
        else:
            start_t = t
        start_t = max(start_t, self.sim_ref_t)

        pulse_cfg = self.pulse_map[name]
        self.pulse_list.append(Pulse(start_t, pulse_cfg))

        if (
            self.sim_last_gen_end_t is None
            or start_t + pulse_cfg["length"] > self.sim_last_gen_end_t
        ):
            self.sim_last_gen_end_t = start_t + pulse_cfg["length"]

    def declare_readout(
        self,
        ch,
        length,
        freq=None,
        phase=0,
        sel="product",
        gen_ch=None,
        edge_counting=False,
        high_threshold=0,
        low_threshold=0,
    ):
        super().declare_readout(
            ch,
            length,
            freq,
            phase,
            sel,
            gen_ch,
            edge_counting,
            high_threshold,
            low_threshold,
        )
        # TODO: this only works for single readout pulse
        self.sim_ro_length = length

    def trigger(
        self, ros=None, pins=None, t=0, width=None, ddr4=False, mr=False, tag=None
    ):
        super().trigger(ros=ros, pins=pins, t=t, width=width, ddr4=ddr4, mr=mr, tag=tag)
        if t is None:
            t = self.sim_ref_t
        t = max(t, self.sim_ref_t)

        # TODO: this only works for single readout pulse
        self.sim_last_ro_end_t = t + self.sim_ro_length

    def visualize(self, time_fly: float = 0.0):
        total_length = max(self.sim_ref_t, self.sim_last_gen_end_t)
        assert total_length is not None, "total_length is None"
        if isinstance(total_length, QickParam):
            total_length = total_length.maxval()

        NUM_SAMPLE = 1001

        times = np.linspace(0.0, total_length, NUM_SAMPLE)
        signal_dict = pulses_to_signal(self, self.pulse_list, times)

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
        slider_names = [list(self.loop_dict.keys())[i] for i in use_dims]
        slider_counts = [self.loop_dict[name] for name in slider_names]

        def plot_func(**slider_vals):
            # 根據滑桿值組合index
            indices = [0] * len(signal_shape)
            indices[-1] = slice(None)
            for i, val in zip(use_dims, slider_vals.values()):
                indices[i] = val
            indices = tuple(indices)

            plt.figure(figsize=(10, 4))
            for ch, sig in signal_dict.items():
                plt.plot(times, np.abs(sig[indices]), label=f"ch {ch}")

            if self.sim_ro_length is not None:
                ro_start = self.sim_last_ro_end_t - self.sim_ro_length
                ro_end = self.sim_last_ro_end_t
                ro_start = format_param(self, ro_start)[indices[:-1]]
                ro_end = format_param(self, ro_end)[indices[:-1]]
                plt.axvline(ro_start - time_fly, color="red", linestyle="--")
                plt.axvline(ro_end - time_fly, color="red", linestyle="--")

            sim_ref_t = format_param(self, self.sim_ref_t)[indices[:-1]]
            plt.axvline(sim_ref_t - time_fly, color="black", linestyle="--")

            plt.xlabel("Time (us)")
            plt.ylabel("Amplitude")
            plt.title(f"{type(self).__name__} Simulation")
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
