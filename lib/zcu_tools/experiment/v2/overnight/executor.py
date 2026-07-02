from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from numpy.typing import NDArray

from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.v2.runner import (
    MeasurementTask,
    MultiMeasurementExecutor,
    Schedule,
)
from zcu_tools.experiment.v2.utils import Result

from .env import OvernightEnv


class OvernightCfg(ExpCfgModel):
    pass


class OvernightExecutor(
    MultiMeasurementExecutor[
        MeasurementTask[OvernightCfg, OvernightEnv, Any, Any, NDArray[np.int64]],
        OvernightCfg,
        OvernightEnv,
        NDArray[np.int64],
    ]
):
    def __init__(self, num_times: int, interval: float) -> None:
        super().__init__()

        if num_times < 0:
            raise ValueError("num_times must be non-negative")
        if interval < 0.0:
            raise ValueError("interval must be non-negative")

        self.num_times = num_times
        self.interval = interval

    @matplotlib.rc_context(
        {"font.size": 6, "xtick.major.size": 6, "ytick.major.size": 6}
    )
    def run(
        self,
        *,
        soc: object,
        soccfg: object,
        fail_retry: int = 3,
    ) -> Mapping[str, Result]:
        iters = np.arange(self.num_times)
        env = OvernightEnv(soc=soc, soccfg=soccfg, iters=iters)
        cfg = OvernightCfg()

        def run_loop(sched: Schedule[OvernightCfg, OvernightEnv]) -> None:
            for _, iter_step in sched.repeat("Iter", self.num_times, self.interval):
                self._run_measurement_batch(iter_step, fail_retry)
                iter_step.trigger_update(flush=True)

        return self._run(
            cfg=cfg,
            env=env,
            outer_values=iters,
            run_loop=run_loop,
        )

    def save(
        self,
        filepath: str,
        results: Mapping[str, Result] | None = None,
        comment: str | None = None,
        prefix_tag: str = "overnight",
    ) -> None:
        if results is None:
            results = self.last_result
        assert results is not None, "no result found"

        iters = np.arange(self.num_times)

        _filepath = Path(filepath)
        for ms_name, ms in self.measurements.items():
            ms.save(
                str(_filepath.with_name(_filepath.name + f"_{ms_name}")),
                iters,
                results[ms_name],
                comment,
                prefix_tag + f"/{ms_name}",
            )
