import warnings
from typing import Tuple

import numpy as np
from numpy import ndarray
from qick.asm_v2 import QickSweep1D

from .twotone import TwoToneProgram


class GEProgram(TwoToneProgram):
    def _initialize(self, cfg):
        # make pi pulse gain 0 or normal
        self.dac["qub_pulse"]["gain"] = QickSweep1D(
            "ge_sweep", 0, self.dac["qub_pulse"]["gain"]
        )
        super()._initialize(cfg)

        # add ge sweep to inner loop
        self.add_loop("ge_sweep", count=2)

    def acquire_shot(self, soc, **kwargs) -> Tuple[ndarray, ndarray]:
        if kwargs.get("soft_avgs", 1) > 1:
            warnings.warn("soft_avgs has no effect in GEProgram.acquire_shot")
            kwargs["soft_avgs"] = 1

        super().acquire(soc, **kwargs)
        acc_buf = self.acc_buf[0]  # type: ignore
        avgiq = acc_buf / self.get_time_axis(0)[-1]  # (reps, 2, 1, 2)
        avgi, avgq = avgiq[..., 0, 0], avgiq[..., 0, 1]  # (reps, 2)
        return avgi.T, avgq.T  # (2, reps)

    def acquire_snr(self, soc, **kwargs):
        avg_d, std2_d = super().acquire(soc, ret_std=True, **kwargs)  # type: ignore
        avg_d = avg_d[0][0].dot([1, 1j])  # type: ignore , (*sweep, ge)
        std2_d = std2_d[0][0].dot([1, 1j])  # type: ignore , (*sweep, ge)

        contrast = avg_d[..., 1] - avg_d[..., 0]  # (*sweep)
        noise2_i = np.sum(std2_d.real**2, axis=-1)  # (*sweep)
        noise2_q = np.sum(std2_d.imag**2, axis=-1)  # (*sweep)
        noise = np.sqrt(
            noise2_i * contrast.real**2 + noise2_q * contrast.imag**2
        ) / np.abs(contrast)

        return contrast / noise

    def acquire(self, soc, **kwargs):
        # default to acquire_snr
        return self.acquire_snr(soc, **kwargs)
