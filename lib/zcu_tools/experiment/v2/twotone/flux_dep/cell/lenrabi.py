from __future__ import annotations

from copy import deepcopy
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np

from zcu_tools.experiment.v2.runner import (
    AbsTask,
    AnalysisTask,
    HardTask,
    ResultType,
    TaskContext,
)
from zcu_tools.experiment.v2.utils import wrap_earlystop_check
from zcu_tools.program.v2 import (
    ModularProgramV2,
    Pulse,
    make_readout,
    make_reset,
    sweep2param,
)
from zcu_tools.utils.fitting import fit_rabi
from zcu_tools.utils.process import rotate2real


class MeasureLenRabiTask(AbsTask):
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

        self.task = HardTask(
            measure_fn=self.measure_fn, result_shape=(len_sweep["expts"],)
        )

    def measure_fn(self, ctx: TaskContext, update_hook: Callable) -> np.ndarray:
        cfg = deepcopy(ctx.cfg)

        fit_freq = ctx.get_data(addr_stack=[*ctx.addr_stack[:-1], "fit_freq"])

        if np.isnan(fit_freq):  # skip if freq measurement failed
            # shape: (ch, ro, lens, iq)
            return [np.full((1, self.len_sweep["expts"], 2), np.nan, dtype=float)]

        cfg["sweep"] = {"length": self.len_sweep}

        snr_hook = None
        if self.snr_ax is not None:

            def snr_hook(snr: float) -> None:
                self.snr_ax.set_title(f"SNR: {snr:.2f}")

        len_params = sweep2param("length", cfg["sweep"]["length"])
        Pulse.set_param(cfg["pi_pulse"], "length", len_params)

        prog = ModularProgramV2(
            self.soccfg,
            cfg,
            modules=[
                make_reset("reset", reset_cfg=cfg.get("reset")),
                Pulse(name="pi_pulse", cfg=cfg["pi_pulse"]),
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


class FitLenRabiTask(AbsTask):
    def __init__(
        self, line_ax: plt.Axes, lens: np.ndarray, singal_key: str = "fit_pi_len"
    ) -> None:
        self.line_ax = line_ax
        self.lens = lens
        self.singal_key = singal_key

        self.line = None
        self.task = AnalysisTask(
            analysis_fn=self.analysis_fn, init_result=np.array(np.nan)
        )

    def analysis_fn(self, ctx: TaskContext) -> np.ndarray:
        freq_signals = ctx.get_data(addr_stack=[*ctx.addr_stack[:-1], self.singal_key])

        real_freq_signals = rotate2real(freq_signals).real
        pi_len, _, _, fit_signals, _ = fit_rabi(
            self.lens, real_freq_signals, decay=True
        )

        # if the fit is not good, return NaN
        if np.mean(np.abs(real_freq_signals - fit_signals)) > 0.3 * np.ptp(
            real_freq_signals
        ):
            return np.nan

        if self.line is None:
            self.line = self.line_ax.axvline(pi_len, color="red", linestyle="--")
        else:
            self.line.set_xdata(pi_len)

        return pi_len

    def init(self, ctx: TaskContext, keep: bool = True) -> None:
        self.task.init(ctx, keep=keep)

    def run(self, ctx: TaskContext) -> None:
        self.task.run(ctx)

    def cleanup(self) -> None:
        self.task.cleanup()

    def get_default_result(self) -> ResultType:
        return self.task.get_default_result()
