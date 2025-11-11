from __future__ import annotations

from copy import deepcopy
from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from zcu_tools.experiment.utils import sweep2array
from zcu_tools.experiment.v2.runner import (
    AbsAutoTask,
    HardTask,
    ResultType,
    TaskContext,
)
from zcu_tools.experiment.v2.utils import set_pulse_freq, wrap_earlystop_check
from zcu_tools.program.v2 import (
    Delay,
    ModularProgramV2,
    Pulse,
    Readout,
    Reset,
    sweep2param,
)
from zcu_tools.utils.fitting import fit_decay, fit_decay_fringe
from zcu_tools.utils.process import rotate2real


def t2ramsey_signal2real(signals: np.ndarray) -> np.ndarray:
    real_signals = np.zeros_like(signals, dtype=np.float64)

    for i in range(signals.shape[0]):
        if np.any(np.isnan(signals[i])):
            continue

        real_signals[i] = rotate2real(signals[i]).real

        # normalize
        max_val = np.max(real_signals[i])
        min_val = np.min(real_signals[i])
        real_signals[i] = (real_signals[i] - min_val) / (max_val - min_val)

    return real_signals


class MeasureT2RamseyTask(AbsAutoTask):
    """
    need: ["qubit_freq", "pi2_length"]
    provide: ["t2ramsey_time"]
    """

    def __init__(
        self,
        soccfg,
        soc,
        len_sweep: dict,
        activate_detune: float = 0.0,
        earlystop_snr: Optional[float] = None,
        plot_ax: Optional[plt.Axes] = None,
    ) -> None:
        self.soccfg = soccfg
        self.soc = soc
        self.len_sweep = len_sweep
        self.activate_detune = activate_detune
        self.earlystop_snr = earlystop_snr
        self.plot_ax = plot_ax

        self.task = HardTask(
            measure_fn=self.measure_t2ramsey_fn, result_shape=(len_sweep["expts"],)
        )

        super().__init__(
            needed_tags=["qubit_freq", "pi2_length"],
            provided_tags=["t2ramsey_time"],
        )

    def measure_t2ramsey_fn(
        self, ctx: TaskContext, update_hook: Callable
    ) -> List[np.ndarray]:
        cfg = deepcopy(ctx.cfg)

        cfg["sweep"] = {"length": self.len_sweep}

        snr_hook = None
        if self.plot_ax is not None:

            def snr_hook(snr: float) -> None:
                self.plot_ax.set_title(f"SNR: {snr:.2f}")

        t2r_params = sweep2param("length", cfg["sweep"]["length"])
        prog = ModularProgramV2(
            self.soccfg,
            cfg,
            modules=[
                Reset("reset", cfg.get("reset", {"type": "none"})),
                Pulse(name="pi2_pulse1", cfg=cfg["pi2_pulse"]),
                Delay(name="t2r_delay", delay=t2r_params),
                Pulse(
                    name="pi2_pulse2",
                    cfg={
                        **ctx.cfg["pi2_pulse"],
                        "phase": ctx.cfg["pi2_pulse"]["phase"]
                        + 360 * self.activate_detune * t2r_params,
                    },
                ),
                Readout("readout", cfg["readout"]),
            ],
        )
        return prog.acquire(
            self.soc,
            progress=False,
            callback=wrap_earlystop_check(
                prog,
                update_hook,
                self.earlystop_snr,
                signal2real_fn=lambda x: rotate2real(x).real,
                snr_hook=snr_hook,
            ),
        )

    def init(self, ctx: TaskContext, keep: bool = True) -> None:
        self.task.init(ctx, keep=keep)

    def run(
        self, ctx: TaskContext, need_infos: Dict[str, complex]
    ) -> Dict[str, complex]:
        fallback_infos = {"t2ramsey_time": np.nan}

        # set pi2 pulse freq to fitted freq
        fit_freq = need_infos.get("qubit_freq", np.nan)
        if np.isnan(fit_freq):
            return fallback_infos
        set_pulse_freq(ctx.cfg["pi2_pulse"], fit_freq)

        # set pi2 pulse length to fitted values
        pi2_len = need_infos.get("pi2_length", np.nan)
        if np.isnan(pi2_len):
            return fallback_infos
        Pulse.set_param(ctx.cfg["pi2_pulse"], "length", pi2_len)

        # measure t2ramsey curve
        self.task.run(ctx)

        # fit t2ramsey time
        lens = sweep2array(self.len_sweep)
        signals = ctx.get_current_data()

        real_signals = rotate2real(signals).real
        if self.activate_detune == 0.0:
            t2ramsey_time, *_ = fit_decay(lens, real_signals)
        else:
            t2ramsey_time, *_ = fit_decay_fringe(lens, real_signals)

        return {"t2ramsey_time": t2ramsey_time}

    def cleanup(self) -> None:
        self.task.cleanup()

    def get_default_result(self) -> ResultType:
        return self.task.get_default_result()
