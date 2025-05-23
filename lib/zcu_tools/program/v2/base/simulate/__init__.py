from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Dict, Type

import numpy as np
from myqick.asm_v2 import QickParam

from ..program import MyProgramV2
from .pulse import Pulse, pulses_to_signal, visualize_pulse
from .waveform import format_param


def update_t(ref_t, t):
    t_a = ref_t
    t_b = t
    if isinstance(t_a, QickParam):
        t_a = 0.5 * (t_a.minval() + t_a.maxval())
    if isinstance(t_b, QickParam):
        t_b = 0.5 * (t_b.minval() + t_b.maxval())
    return t if t_b > t_a else ref_t


class HookWrapper:
    def __init__(self, wrapped_type: Type, method_hooks: Dict[str, Callable]):
        self.wrapped_type = wrapped_type
        self.method_hooks = method_hooks
        self.orig_methods: Dict[str, Callable] = {}

    def __enter__(self) -> None:
        for name, hook in self.method_hooks.items():
            orig_method = getattr(self.wrapped_type, name)
            self.orig_methods[name] = orig_method

            def make_wrapper(orig_method, hook):
                @wraps(orig_method)
                def wrapped_method(self, *args, **kwargs) -> Any:
                    result = orig_method(self, *args, **kwargs)
                    hook(*args, **kwargs)  # don't need self
                    return result

                return wrapped_method

            new_method = make_wrapper(orig_method, hook)

            setattr(self.wrapped_type, name, new_method)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for name, orig_method in self.orig_methods.items():
            setattr(self.wrapped_type, name, orig_method)

        self.orig_methods.clear()


class SimulateProgramV2:
    """
    Record the pulse sequence in a list of Pulse objects, So we can plot them later.
    It is performed by overriding the delay and pulse methods.
    It isn't very accurate, but it is enough for most cases.
    """

    def __init__(
        self, prog_type: Type[MyProgramV2], soccfg: Dict[str, Any], cfg: Dict[str, Any]
    ) -> None:
        self.sim_ref_t = 0.0

        self.sim_last_gen_end_t = None
        self.sim_last_ro_end_t = None

        self.pulse_list = []
        self.pulse_map = dict()
        self.sim_ro_length = None

        self.prog_type = prog_type

        method_hooks = {
            "declare_pulse": self.declare_pulse_hook,
            "delay": self.delay_pulse_hook,
            "delay_auto": self.delay_auto_hook,
            "pulse": self.pulse_hook,
            "declare_readout": self.declare_readout_hook,
            "trigger": self.trigger_hook,
        }
        with HookWrapper(prog_type, method_hooks):
            self.prog = prog_type(soccfg, cfg)

    def declare_pulse_hook(self, pulse, name, *args, **kwargs) -> None:
        self.pulse_map[(pulse["ch"], name)] = deepcopy(pulse)

    def delay_pulse_hook(self, t, *args, **kwargs) -> None:
        self.sim_ref_t = update_t(self.sim_ref_t, t)

    def delay_auto_hook(self, t=0, gens=True, ros=True, *args, **kwargs) -> None:
        last_end = self.sim_ref_t
        if gens and self.sim_last_gen_end_t is not None:
            last_end = update_t(last_end, self.sim_last_gen_end_t)
        if ros and self.sim_last_ro_end_t is not None:
            last_end = update_t(last_end, self.sim_last_ro_end_t)
        self.sim_ref_t = update_t(self.sim_ref_t, last_end + t)

    def pulse_hook(self, ch, name, t=0, *args, **kwargs) -> None:
        start_t = 0.0
        if t == "auto":
            if self.sim_last_gen_end_t is not None:
                start_t = self.sim_last_gen_end_t
        else:
            start_t = t
        start_t = update_t(start_t, self.sim_ref_t)

        pulse_cfg = self.pulse_map[(ch, name)]
        self.pulse_list.append(Pulse(start_t, pulse_cfg))

        pulse_end = start_t + pulse_cfg["length"]
        if self.sim_last_gen_end_t is None:
            self.sim_last_gen_end_t = pulse_end
        else:
            self.sim_last_gen_end_t = update_t(self.sim_last_gen_end_t, pulse_end)

    def declare_readout_hook(self, ch, length, *args, **kwargs) -> None:
        # TODO: this only works for single readout pulse
        self.sim_ro_length = length

    def trigger_hook(self, ros=None, pins=None, t=0, *args, **kwargs) -> None:
        if t is None:
            t = self.sim_ref_t
        t = update_t(t, self.sim_ref_t)

        # TODO: this only works for single readout pulse
        self.sim_last_ro_end_t = t + self.sim_ro_length

    def visualize(self, time_fly: float = 0.0) -> None:
        total_length = update_t(
            self.sim_ref_t,
            0.0 if self.sim_last_gen_end_t is None else self.sim_last_gen_end_t,
        )
        assert total_length is not None, "total_length is None"
        if isinstance(total_length, QickParam):
            total_length = total_length.maxval()

        NUM_SAMPLE = 1001

        visualize_keywords = ["length", "sigma", "alpha", "gain"]
        loop_dict = {
            k: v
            for k, v in self.prog.loop_dict.items()
            if any(kw in k for kw in visualize_keywords)
        }

        times = np.linspace(0.0, total_length, NUM_SAMPLE)
        signal_dict = pulses_to_signal(loop_dict, self.pulse_list, times)

        # remove unused dimensions

        import ipywidgets as widgets
        import matplotlib.pyplot as plt
        from IPython.display import display

        seq_lengths = format_param(loop_dict, self.sim_ref_t)
        if self.sim_ro_length is not None:
            ro_start = self.sim_last_ro_end_t - self.sim_ro_length
            ro_end = self.sim_last_ro_end_t
            ro_start = format_param(loop_dict, ro_start)
            ro_end = format_param(loop_dict, ro_end)

        def plot_func(plot_type="abs", **slider_vals) -> None:
            nonlocal seq_lengths, times, signal_dict, loop_dict

            idxs = tuple(slider_vals.values())

            plt.figure(figsize=(10, 4))
            for ch, sig in signal_dict.items():
                sig = sig[idxs]
                if plot_type == "abs":
                    plt.plot(times, np.abs(sig), label=f"ch {ch}")
                elif plot_type == "real/imag":
                    plt.plot(times, np.real(sig), label=f"ch {ch} real")
                    plt.plot(times, np.imag(sig), label=f"ch {ch} imag")
                else:
                    raise ValueError(f"Invalid plot type: {plot_type}")

            if self.sim_ro_length is not None:
                plt.axvline(ro_start[idxs] - time_fly, color="red", linestyle="--")
                plt.axvline(ro_end[idxs] - time_fly, color="red", linestyle="--")

            plt.axvline(seq_lengths[idxs], color="black", linestyle="--")

            all_sig = np.stack(list(signal_dict.values()), axis=-1)
            if plot_type == "abs":
                plt.ylim(0.0, 1.1 * np.max(np.abs(all_sig)))
            elif plot_type == "real/imag":
                plt.ylim(
                    1.1 * min(np.min(np.real(all_sig)), np.min(np.imag(all_sig)), 0.0),
                    1.1 * max(np.max(np.real(all_sig)), np.max(np.imag(all_sig)), 0.0),
                )
            plt.xlabel("Time (us)")
            plt.ylabel("Amplitude")
            plt.title(f"{self.prog_type.__name__} Simulation")
            plt.legend()
            plt.grid(True)
            plt.show()

        # 根據use_indices判斷哪些維度需要滑桿
        slider_names = list(loop_dict.keys())
        slider_counts = list(loop_dict.values())

        plot_type_dropdown = widgets.Dropdown(
            options=["abs", "real/imag"],
            value="abs",
            description="Plot Type:",
        )

        if len(slider_names) > 0:
            sliders = [
                widgets.IntSlider(min=0, max=c - 1, step=1, description=n)
                for n, c in zip(slider_names, slider_counts)
            ]
            slider_box = widgets.HBox(sliders)
            ui = widgets.VBox([plot_type_dropdown, slider_box])
            interactive_widgets = {n: s for n, s in zip(slider_names, sliders)}
            interactive_widgets["plot_type"] = plot_type_dropdown
            out = widgets.interactive_output(plot_func, interactive_widgets)
            display(ui, out)
        else:
            # Even if there are no sliders, we still want the plot type dropdown
            ui = widgets.VBox([plot_type_dropdown])
            out = widgets.interactive_output(
                plot_func, {"plot_type": plot_type_dropdown}
            )
            display(ui, out)


__all__ = ["SimulateProgramV2", "visualize_pulse"]
