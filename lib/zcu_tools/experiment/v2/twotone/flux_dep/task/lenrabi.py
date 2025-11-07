from __future__ import annotations

from copy import deepcopy
from typing import Callable, Dict, Optional

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
    ModularProgramV2,
    Pulse,
    make_readout,
    make_reset,
    sweep2param,
)
from zcu_tools.utils.fitting import fit_rabi
from zcu_tools.utils.process import rotate2real


def lenrabi_signal2real(signals: np.ndarray) -> np.ndarray:
    real_signals = np.full_like(signals, np.nan, dtype=np.float64)

    for i in range(signals.shape[0]):
        if np.any(np.isnan(signals[i])):
            continue

        real_signals[i] = rotate2real(signals[i]).real

        # normalize
        max_val = np.max(real_signals[i])
        min_val = np.min(real_signals[i])
        real_signals[i] = (real_signals[i] - min_val) / (max_val - min_val)

    return real_signals


class MeasureLenRabiTask(AbsAutoTask):
    """
    need: ["qubit_freq"]
    provide: ["pi_length", "pi2_length", "rabi_freq"]
    """

    def __init__(
        self,
        soccfg,
        soc,
        len_sweep: dict,
        pulse_name: str = "pi_pulse",
        earlystop_snr: Optional[float] = None,
    ) -> None:
        self.soccfg = soccfg
        self.soc = soc
        self.len_sweep = len_sweep
        self.pulse_name = pulse_name
        self.earlystop_snr = earlystop_snr
        self.task = HardTask(
            measure_fn=self.measure_lenrabi_fn, result_shape=(len_sweep["expts"],)
        )

        super().__init__(
            needed_tags=["qubit_freq"],
            provided_tags=["pi_length", "pi2_length", "rabi_freq"],
        )

    def measure_lenrabi_fn(self, ctx: TaskContext, update_hook: Callable) -> np.ndarray:
        cfg = deepcopy(ctx.cfg)

        cfg["sweep"] = {"length": self.len_sweep}

        len_params = sweep2param("length", cfg["sweep"]["length"])
        Pulse.set_param(cfg[self.pulse_name], "length", len_params)

        prog = ModularProgramV2(
            self.soccfg,
            cfg,
            modules=[
                make_reset("reset", reset_cfg=cfg.get("reset")),
                Pulse(name=self.pulse_name, cfg=cfg[self.pulse_name]),
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
            ),
        )

    def init(self, ctx: TaskContext, keep: bool = True) -> None:
        self.task.init(ctx, keep=keep)

    def run(self, ctx: TaskContext, need_infos: Dict[str, np.ndarray]) -> None:
        fallback_infos = {
            "pi_length": np.nan,
            "pi2_length": np.nan,
            "rabi_freq": np.nan,
        }

        # set pulse freq to fitted freq
        fit_freq = need_infos.get("qubit_freq", np.nan)
        if np.isnan(fit_freq):
            return fallback_infos
        set_pulse_freq(ctx.cfg[self.pulse_name], fit_freq)

        # measure len rabi
        self.task.run(ctx)

        # fit pi and pi2 len
        lens = sweep2array(self.len_sweep)
        signals = ctx.get_current_data()

        real_signals = rotate2real(signals).real
        pi_len, pi2_len, rabi_freq, fit_signals, _ = fit_rabi(
            lens, real_signals, decay=True
        )

        # if the fit is not good, set all results to NaN
        if np.mean(np.abs(real_signals - fit_signals)) > 0.1 * np.ptp(real_signals):
            pi_len = np.nan
            pi2_len = np.nan
            rabi_freq = np.nan

        return dict(
            pi_length=pi_len,
            pi2_length=pi2_len,
            rabi_freq=rabi_freq,
        )

    def cleanup(self) -> None:
        self.task.cleanup()

    def get_default_result(self) -> ResultType:
        return self.task.get_default_result()
