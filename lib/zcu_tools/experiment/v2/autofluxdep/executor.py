from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from zcu_tools.device import DeviceInfo
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import set_flux_in_dev_cfg, setup_devices
from zcu_tools.experiment.v2.runner import (
    MeasurementTask,
    MultiMeasurementExecutor,
    Schedule,
)
from zcu_tools.experiment.v2.utils import Result
from zcu_tools.meta_tool import ModuleLibrary
from zcu_tools.simulate.fluxonium import FluxoniumPredictor

from .env import FluxDepEnv, FluxDepInfoTracker


class FluxDepCfg(ExpCfgModel):
    # Field(...) makes dev required in this subclass, overriding the Optional
    # default from ExpCfgModel — intentional Pydantic pattern (type: ignore[override]).
    dev: dict[str, DeviceInfo] = Field(...)  # type: ignore[override]


class FluxDepExecutor(
    MultiMeasurementExecutor[
        MeasurementTask[FluxDepCfg, FluxDepEnv, Any, Any, NDArray[np.float64]],
        FluxDepCfg,
        FluxDepEnv,
        NDArray[np.float64],
    ]
):
    def __init__(self, flux_values: NDArray[np.float64]) -> None:
        super().__init__()

        self.flux_values = flux_values

    @matplotlib.rc_context(
        {"font.size": 6, "xtick.major.size": 6, "ytick.major.size": 6}
    )
    def run(
        self,
        dev_cfg: dict[str, DeviceInfo],
        predictor: FluxoniumPredictor,
        *,
        soc: object,
        soccfg: object,
        ml: ModuleLibrary,
        retry_time: int = 3,
    ) -> Mapping[str, Result]:
        cfg = FluxDepCfg(dev=dev_cfg)
        env = FluxDepEnv(
            soc=soc,
            soccfg=soccfg,
            ml=ml,
            flux_values=self.flux_values,
            predictor=predictor,
            info=FluxDepInfoTracker(),
        )

        set_flux_in_dev_cfg(cfg.dev, self.flux_values[0], label="flux_dev")
        setup_devices(cfg, progress=True)

        def run_loop(sched: Schedule[FluxDepCfg, FluxDepEnv]) -> None:
            for i, (flux, flux_step) in enumerate(sched.scan("flux", self.flux_values)):
                info = flux_step.env.info
                predictor = flux_step.env.predictor

                cur_m = predictor.predict_matrix_element(flux)
                first_m = info.first.cur_m if info.first.cur_m is not None else cur_m
                info.start_step(
                    flux_value=float(flux),
                    flux_idx=i,
                    cur_m=float(cur_m),
                    m_ratio=float(cur_m / first_m),
                )

                set_flux_in_dev_cfg(flux_step.cfg.dev, flux, label="flux_dev")
                self._run_measurement_batch(flux_step, retry_time)

        return self._run(
            cfg=cfg,
            env=env,
            outer_values=self.flux_values,
            retry_time=retry_time,
            run_loop=run_loop,
        )

    def save(
        self,
        filepath: str,
        result: Mapping[str, Result] | None = None,
        comment: str | None = None,
        prefix_tag: str = "autoflux_dep",
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        _filepath = Path(filepath)
        for ms_name, ms in self.measurements.items():
            ms.save(
                str(_filepath.with_name(_filepath.name + f"_{ms_name}")),
                self.flux_values,
                result[ms_name],
                comment,
                prefix_tag + f"/{ms_name}",
            )
