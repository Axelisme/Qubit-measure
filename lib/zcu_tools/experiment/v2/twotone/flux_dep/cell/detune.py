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
from zcu_tools.utils.fitting import fit_qubit_freq
from zcu_tools.utils.process import minus_background, rotate2real


class MeasureDetuneTask(AbsTask):
    def __init__(
        self,
        soccfg,
        soc,
        detune_sweep: dict,
        earlystop_snr: Optional[float] = None,
        snr_ax: Optional[plt.Axes] = None,
    ) -> None:
        self.soccfg = soccfg
        self.soc = soc
        self.detune_sweep = detune_sweep
        self.earlystop_snr = earlystop_snr
        self.snr_ax = snr_ax

        self.task = HardTask(
            measure_fn=self.measure_fn, result_shape=(len(detune_sweep),)
        )

    def measure_fn(self, ctx: TaskContext, update_hook: Callable) -> np.ndarray:
        cfg = deepcopy(ctx.cfg)

        cfg["sweep"] = {"detune": self.detune_sweep}
        cfg["relax_delay"] = 1.0  # no need for freq measurement

        detune_params = sweep2param("detune", cfg["sweep"]["detune"])

        snr_hook = None
        if self.snr_ax is not None:

            def snr_hook(snr: float) -> None:
                self.snr_ax.set_title(f"SNR: {snr:.2f}")

        prog = ModularProgramV2(
            self.soccfg,
            cfg,
            modules=[
                make_reset("reset", reset_cfg=cfg.get("reset")),
                Pulse(
                    name="qub_pulse",
                    cfg={
                        **cfg["qub_pulse"],
                        "freq": cfg["qub_pulse"]["freq"] + detune_params,
                    },
                ),
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


class FitLastFreqTask(AbsTask):
    def __init__(
        self, line_ax: plt.Axes, detunes: np.ndarray, singal_key: str = "detune"
    ) -> None:
        self.line_ax = line_ax
        self.detunes = detunes
        self.singal_key = singal_key

        self.line = None
        self.task = AnalysisTask(
            analysis_fn=self.analysis_fn, init_result=np.array(np.nan)
        )

    def analysis_fn(self, ctx: TaskContext) -> np.ndarray:
        freq_signals = ctx.get_data(addr_stack=[*ctx.addr_stack[:-1], self.singal_key])

        real_freq_signals = np.abs(minus_background(freq_signals))
        detune, freq_err, kappa, *_ = fit_qubit_freq(self.detunes, real_freq_signals)
        if freq_err > 0.5 * kappa:
            return np.nan  # fit failed
        else:
            if self.line is None:
                self.line = self.line_ax.axvline(detune, color="red", linestyle="--")
            else:
                self.line.set_xdata(detune)

            return detune + ctx.cfg["qub_pulse"]["freq"]

    def init(self, ctx: TaskContext, keep: bool = True) -> None:
        self.task.init(ctx, keep=keep)

    def run(self, ctx: TaskContext) -> None:
        self.task.run(ctx)

    def cleanup(self) -> None:
        self.task.cleanup()

    def get_default_result(self) -> ResultType:
        return self.task.get_default_result()
