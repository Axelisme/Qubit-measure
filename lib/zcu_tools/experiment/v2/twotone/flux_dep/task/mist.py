from __future__ import annotations

from copy import deepcopy
from typing import Callable, Dict, List

import numpy as np

from zcu_tools.experiment.v2.runner import (
    AbsAutoTask,
    HardTask,
    ResultType,
    TaskContext,
)
from zcu_tools.experiment.v2.utils import set_pulse_freq
from zcu_tools.program.v2 import (
    ModularProgramV2,
    Pulse,
    make_readout,
    make_reset,
    sweep2param,
)


def mist_signal2real(signals: np.ndarray) -> np.ndarray:
    g_signals, e_signals = signals[..., 0], signals[..., 1]  # (flxs, pdrs, ge)

    avg_len = max(int(0.05 * g_signals.shape[1]), 1)

    sum_signals = e_signals + g_signals
    mist_signals = sum_signals - np.mean(
        sum_signals[:, :avg_len], axis=1, keepdims=True
    )

    return np.abs(mist_signals)


class MeasureMistTask(AbsAutoTask):
    """need: ["qubit_freq", "pi_length"]"""

    def __init__(
        self, soccfg, soc, pdr_sweep: dict, pulse_name: str = "probe_pulse"
    ) -> None:
        self.soccfg = soccfg
        self.soc = soc
        self.pdr_sweep = pdr_sweep
        self.pulse_name = pulse_name

        self.task = HardTask(
            measure_fn=self.measure_mist_fn, result_shape=(pdr_sweep["expts"], 2)
        )

        super().__init__(needed_tags=["qubit_freq", "pi_length"])

    def measure_mist_fn(
        self, ctx: TaskContext, update_hook: Callable
    ) -> List[np.ndarray]:
        cfg = deepcopy(ctx.cfg)

        cfg["sweep"] = {
            "gain": self.pdr_sweep,
            "ge": {"start": 0, "stop": cfg["pi_pulse"]["gain"], "expts": 2},
        }

        pdr_params = sweep2param("gain", cfg["sweep"]["gain"])
        ge_params = sweep2param("ge", cfg["sweep"]["ge"])
        Pulse.set_param(cfg["pi_pulse"], "on/off", ge_params)
        Pulse.set_param(cfg["probe_pulse"], "gain", pdr_params)

        return ModularProgramV2(
            self.soccfg,
            cfg,
            modules=[
                make_reset("reset", reset_cfg=cfg.get("reset")),
                Pulse(name="probe_pulse", cfg=cfg[self.pulse_name]),
                Pulse(name="pi_pulse", cfg=cfg["pi_pulse"]),
                make_readout("readout", readout_cfg=cfg["readout"]),
            ],
        ).acquire(self.soc, progress=False, callback=update_hook)

    def init(self, ctx: TaskContext, keep: bool = True) -> None:
        self.task.init(ctx, keep=keep)

    def run(self, ctx: TaskContext, need_infos: Dict[str, np.ndarray]) -> None:
        fallback_infos = {}

        # set pulse freq to fitted freq
        fit_freq = need_infos.get("qubit_freq", np.nan)
        if np.isnan(fit_freq):
            return fallback_infos
        set_pulse_freq(ctx.cfg[self.pulse_name], fit_freq)
        set_pulse_freq(ctx.cfg["pi_pulse"], fit_freq)

        # optional: set pi pulse length to fitted value
        fit_pi_len = need_infos.get("pi_length", np.nan)
        if np.isnan(fit_pi_len):
            return fallback_infos
        Pulse.set_param(ctx.cfg["pi_pulse"], "length", fit_pi_len)

        # measure mist
        self.task.run(ctx)

        return {}

    def cleanup(self) -> None:
        self.task.cleanup()

    def get_default_result(self) -> ResultType:
        return self.task.get_default_result()
