from __future__ import annotations

from copy import deepcopy
from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from zcu_tools.experiment.utils import sweep2array
from zcu_tools.experiment.v2.utils import wrap_earlystop_check
from zcu_tools.program.v2 import (
    ModularProgramV2,
    Pulse,
    make_readout,
    make_reset,
    sweep2param,
)
from zcu_tools.utils.fitting import fit_qubit_freq
from zcu_tools.utils.process import rotate2real

from zcu_tools.experiment.v2.runner import (
    HardTask,
    ResultType,
    TaskContext,
    AbsAutoTask,
)


def detune_signal2real(signals: np.ndarray) -> np.ndarray:
    real_signals = np.zeros_like(signals, dtype=np.float64)

    flx_len = signals.shape[0]
    for i in range(flx_len):
        real_signals[i, :] = rotate2real(signals[i : min(i + 1, flx_len), :]).real[0]

        if np.any(np.isnan(real_signals[i, :])):
            continue

        # flip to peak up
        max_val = np.max(real_signals[i, :])
        min_val = np.min(real_signals[i, :])
        avg_val = np.mean(real_signals[i, :])
        if max_val + min_val < 2 * avg_val:
            real_signals[i, :] = -real_signals[i, :]
            max_val, min_val = -min_val, -max_val

        # normalize
        real_signals[i, :] = (real_signals[i, :] - min_val) / (max_val - min_val)

    return real_signals


class MeasureDetuneTask(AbsAutoTask):
    """provide: ["qubit_freq"]"""

    def __init__(
        self,
        soccfg,
        soc,
        detune_sweep: dict,
        earlystop_snr: Optional[float] = None,
        plot_ax: Optional[plt.Axes] = None,
    ) -> None:
        self.soccfg = soccfg
        self.soc = soc
        self.detune_sweep = detune_sweep
        self.earlystop_snr = earlystop_snr
        self.plot_ax = plot_ax

        self.freq_line = None
        self.task = HardTask(
            measure_fn=self.measure_freq_fn, result_shape=(detune_sweep["expts"],)
        )

        super().__init__(provided_tags=["qubit_freq"])

    def measure_freq_fn(
        self, ctx: TaskContext, update_hook: Callable
    ) -> List[np.ndarray]:
        cfg = deepcopy(ctx.cfg)

        cfg["sweep"] = {"detune": self.detune_sweep}
        cfg["relax_delay"] = 1.0  # no need for freq measurement

        detune_params = sweep2param("detune", cfg["sweep"]["detune"])

        snr_hook = None
        if self.plot_ax is not None:

            def snr_hook(snr: float) -> None:
                self.plot_ax.set_title(f"SNR: {snr:.2f}")

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

    def run(
        self, ctx: TaskContext, need_infos: Dict[str, complex]
    ) -> Dict[str, complex]:
        self.task.run(ctx)

        detunes = sweep2array(self.detune_sweep)
        signals = ctx.get_current_data()

        real_signals = rotate2real(signals).real
        detune, freq_err, kappa, *_ = fit_qubit_freq(detunes, real_signals)
        if freq_err > 0.3 * kappa:
            fit_freq = np.nan
        else:
            fit_freq = detune + ctx.cfg["qub_pulse"]["freq"]

            # update frequency line
            if self.plot_ax is not None:
                if self.freq_line is None:
                    self.freq_line = self.plot_ax.axvline(
                        detune, color="red", linestyle="--"
                    )
                else:
                    self.freq_line.set_xdata(detune)

        return {"qubit_freq": fit_freq}

    def cleanup(self) -> None:
        self.task.cleanup()

    def get_default_result(self) -> ResultType:
        return self.task.get_default_result()
