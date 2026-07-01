from __future__ import annotations

from abc import ABC, abstractmethod
from collections import UserDict
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Generic, TypeVar

import matplotlib
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from pydantic import Field

from zcu_tools.device import DeviceInfo
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import set_flux_in_dev_cfg, setup_devices
from zcu_tools.experiment.v2.runner.multi_executor import (
    MeasurementContext,
    MultiMeasurementExecutor,
)
from zcu_tools.experiment.v2.utils import Result, merge_result_list
from zcu_tools.liveplot import AbsLivePlot
from zcu_tools.progress_bar import make_pbar
from zcu_tools.simulate.fluxonium import FluxoniumPredictor

T_PlotDict = TypeVar("T_PlotDict", bound=Mapping[str, AbsLivePlot])


T_Result = TypeVar("T_Result", bound=Result)
T_RootResult = TypeVar("T_RootResult", bound=Result)


class FluxDepCfg(ExpCfgModel):
    # Field(...) makes dev required in this subclass, overriding the Optional
    # default from ExpCfgModel — intentional Pydantic pattern (type: ignore[override]).
    dev: dict[str, DeviceInfo] = Field(...)  # type: ignore[override]


class MeasurementTask(
    ABC,
    Generic[T_Result, T_RootResult, T_PlotDict],
):
    def init(self, dynamic_pbar: bool = False) -> None: ...

    @abstractmethod
    def run(
        self, state: MeasurementContext[T_Result, T_RootResult, FluxDepCfg]
    ) -> None: ...

    def cleanup(self) -> None: ...

    @abstractmethod
    def get_default_result(self) -> T_Result: ...

    @abstractmethod
    def num_axes(self) -> dict[str, int]: ...

    @abstractmethod
    def make_plotter(self, name: str, axs: dict[str, list[Axes]]) -> T_PlotDict: ...

    @abstractmethod
    def update_plotter(
        self,
        plotters: T_PlotDict,
        ctx: MeasurementContext[Any, Any, Any],
        signals: T_Result,
    ) -> None: ...

    @abstractmethod
    def save(
        self,
        filepath: str,
        flux_values: NDArray[np.float64],
        result: T_Result,
        comment: str | None,
        prefix_tag: str,
    ) -> None: ...


class FluxDepInfoDict(UserDict):
    def __init__(self, initialdata: Mapping[str, Any] | None = None) -> None:
        self.first_info: dict[str, Any] = {}
        self.last_info: dict[str, Any] = {}
        super().__init__(initialdata)

    @property
    def last(self) -> dict[str, Any]:
        return self.last_info

    @property
    def first(self) -> dict[str, Any]:
        return self.first_info

    def __setitem__(self, key: str, item: Any) -> None:
        super().__setitem__(key, item)
        self.first_info.setdefault(key, deepcopy(item))
        self.last_info[key] = deepcopy(item)


class FluxDepExecutor(MultiMeasurementExecutor[MeasurementTask, FluxDepCfg]):
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
        env_dict: dict[str, Any] | None = None,
        retry_time: int = 3,
    ) -> Mapping[str, Result]:
        if len(self.measurements) == 0:
            raise ValueError("No measurements added")

        if env_dict is None:
            env_dict = {}

        cfg = FluxDepCfg(dev=dev_cfg)

        env_dict.update(
            flux_values=self.flux_values,
            predictor=predictor,
            info=FluxDepInfoDict(),
        )

        def update_flux_context(
            i: int,
            ctx: MeasurementContext[
                dict[str, Result], list[dict[str, Result]], FluxDepCfg
            ],
            flux: float,
        ) -> None:
            info: FluxDepInfoDict = ctx.env["info"]
            predictor: FluxoniumPredictor = ctx.env["predictor"]

            info.clear()  # clear current info dict

            info["flux_value"] = flux
            info["flux_idx"] = i

            info["cur_m"] = predictor.predict_matrix_element(flux)
            info["m_ratio"] = info["cur_m"] / info.first["cur_m"]

            set_flux_in_dev_cfg(ctx.cfg.dev, flux, label="flux_dev")

        set_flux_in_dev_cfg(cfg.dev, self.flux_values[0], label="flux_dev")
        setup_devices(cfg, progress=True)

        init_result = [
            self._default_batch_result() for _ in range(len(self.flux_values))
        ]

        def run_loop(
            root_ctx: MeasurementContext[
                list[dict[str, Result]], list[dict[str, Result]], FluxDepCfg
            ],
        ) -> None:
            pbar = make_pbar(
                total=len(self.flux_values), smoothing=0, desc="flux", leave=True
            )
            try:
                for i, flux in enumerate(self.flux_values.tolist()):
                    if root_ctx.is_stop():
                        break
                    flux_ctx = root_ctx.child(i)
                    update_flux_context(i, flux_ctx, flux)

                    for name, measurement in self.measurements.items():
                        if root_ctx.is_stop():
                            break
                        measurement_ctx = flux_ctx.child(name)
                        self._run_measurement_with_retries(
                            measurement, measurement_ctx, retry_time
                        )
                    pbar.update()
            finally:
                pbar.close()

        results = self._run_with_plotting(init_result, cfg, env_dict, run_loop)

        signals_dict = merge_result_list(results)

        self.last_cfg = cfg
        self.last_result = signals_dict

        return signals_dict

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
