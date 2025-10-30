from __future__ import annotations

from copy import deepcopy
from typing import Callable, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from zcu_tools.experiment.v2.runner import (
    AbsTask,
    HardTask,
    ResultType,
    TaskContext,
)
from zcu_tools.experiment.v2.utils import set_pulse_freq, wrap_earlystop_check
from zcu_tools.program.v2 import (
    Delay,
    ModularProgramV2,
    Pulse,
    make_readout,
    make_reset,
    sweep2param,
)
from zcu_tools.utils.process import rotate2real


class MeasureT1Task(AbsTask):
    def __init__(
        self,
        soccfg,
        soc,
        len_sweep: dict,
        earlystop_snr: Optional[float] = None,
        snr_ax: Optional[plt.Axes] = None,
    ) -> None:
        self.soccfg = soccfg
        self.soc = soc
        self.len_sweep = len_sweep
        self.earlystop_snr = earlystop_snr
        self.snr_ax = snr_ax

        self.task = HardTask(measure_fn=self.measure_fn, result_shape=(len(len_sweep),))

    def measure_fn(self, ctx: TaskContext, update_hook: Callable) -> List[np.ndarray]:
        cfg = deepcopy(ctx.cfg)

        fit_freq = ctx.get_data(addr_stack=[*ctx.addr_stack[:-1], "fit_freq"])
        cfg["sweep"] = {"length": self.len_sweep}

        if np.isnan(fit_freq):  # skip if freq measurement failed
            # shape: (ch, ro, lens, iq)
            return [np.full((1, self.len_sweep["expts"], 2), np.nan, dtype=float)]

        set_pulse_freq(cfg["pi_pulse"], fit_freq)
        t1_span = sweep2param("length", cfg["sweep"]["length"])

        snr_hook = None
        if self.snr_ax is not None:

            def snr_hook(snr: float) -> None:
                self.snr_ax.set_title(f"SNR: {snr:.2f}")

        prog = ModularProgramV2(
            self.soccfg,
            cfg,
            modules=[
                make_reset("reset", reset_cfg=cfg.get("reset")),
                Pulse(name="pi_pulse", cfg=cfg["pi_pulse"]),
                Delay(name="t1_delay", delay=t1_span),
                make_readout("readout", readout_cfg=cfg["readout"]),
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

    def run(self, ctx: TaskContext) -> None:
        self.task.run(ctx)

    def cleanup(self) -> None:
        self.task.cleanup()

    def get_default_result(self) -> ResultType:
        return self.task.get_default_result()
