import warnings
from typing import Tuple

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

    def acquire(self, soc, **kwargs):
        IQlist = super().acquire(soc, **kwargs)
        return [iq[..., 1, :] - iq[..., 0, :] for iq in IQlist]  # type: ignore
