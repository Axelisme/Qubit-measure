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
from zcu_tools.utils.fitting import fit_decay
from zcu_tools.utils.process import rotate2real


def t1_signal2real(signals: np.ndarray) -> np.ndarray:
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


class MeasureT1Task(AbsAutoTask):
    """
    need: ["qubit_freq", "pi_length"]
    provide: ["t1_time"]
    """

    def __init__(
        self,
        soccfg,
        soc,
        len_sweep: dict,
        earlystop_snr: Optional[float] = None,
        plot_ax: Optional[plt.Axes] = None,
    ) -> None:
        self.soccfg = soccfg
        self.soc = soc
        self.len_sweep = len_sweep
        self.earlystop_snr = earlystop_snr
        self.plot_ax = plot_ax

        self.task = HardTask(
            measure_fn=self.measure_t1_fn, result_shape=(len_sweep["expts"],)
        )

        super().__init__(
            needed_tags=["qubit_freq", "pi_length"],
            provided_tags=["t1_time"],
        )

    def measure_t1_fn(
        self, ctx: TaskContext, update_hook: Callable
    ) -> List[np.ndarray]:
        cfg = deepcopy(ctx.cfg)

        cfg["sweep"] = {"length": self.len_sweep}

        snr_hook = None
        if self.plot_ax is not None:

            def snr_hook(snr: float) -> None:
                self.plot_ax.set_title(f"SNR: {snr:.2f}")

        prog = ModularProgramV2(
            self.soccfg,
            cfg,
            modules=[
                Reset("reset", cfg.get("reset", {"type": "none"})),
                Pulse(name="pi_pulse", cfg=cfg["pi_pulse"]),
                Delay(
                    name="t1_delay", delay=sweep2param("length", cfg["sweep"]["length"])
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
        fallback_infos = {"t1_time": np.nan}

        # set pi pulse freq to fitted freq
        fit_freq = need_infos.get("qubit_freq", np.nan)
        if np.isnan(fit_freq):
            return fallback_infos
        set_pulse_freq(ctx.cfg["pi_pulse"], fit_freq)

        # set pi pi pulse length to fitted values
        pi_len = need_infos.get("pi_length", np.nan)
        if np.isnan(pi_len):
            return fallback_infos
        Pulse.set_param(ctx.cfg["pi_pulse"], "length", pi_len)

        # measure t1 curve
        self.task.run(ctx)

        # fit t1 time
        lens = sweep2array(self.len_sweep)
        signals = ctx.get_current_data()
        real_signals = rotate2real(signals).real
        t1_time, *_ = fit_decay(lens, real_signals)

        return {"t1_time": t1_time}

    def cleanup(self) -> None:
        self.task.cleanup()

    def get_default_result(self) -> ResultType:
        return self.task.get_default_result()
