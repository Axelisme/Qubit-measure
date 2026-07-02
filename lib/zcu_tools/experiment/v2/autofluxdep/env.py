from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
from numpy.typing import NDArray

from zcu_tools.meta_tool import ModuleLibrary
from zcu_tools.simulate.fluxonium import FluxoniumPredictor


@dataclass(slots=True)
class FluxDepInfo:
    flux_value: float | None = None
    flux_idx: int | None = None
    cur_m: float | None = None
    m_ratio: float | None = None
    predict_freq: float | None = None
    qubit_freq: float | None = None
    fit_detune: float | None = None
    fit_kappa: float | None = None
    qfw_factor: float | None = None
    qubfreq_success_idx: int | None = None
    t1: float | None = None
    smooth_t1: float | None = None
    t2r: float | None = None
    t2r_detune: float | None = None
    smooth_t2r: float | None = None
    t2e: float | None = None
    smooth_t2e: float | None = None
    pi_length: float | None = None
    pi2_length: float | None = None
    pi_pulse: Any | None = None
    pi2_pulse: Any | None = None
    smooth_pi_product: float | None = None
    lenrabi_success_idx: int | None = None
    best_ro_freq: float | None = None
    best_ro_gain: float | None = None
    opt_readout: Any | None = None


class FluxDepInfoTracker:
    def __init__(self) -> None:
        self.current = FluxDepInfo()
        self.first = FluxDepInfo()
        self.last = FluxDepInfo()
        self._field_names = {field.name for field in fields(FluxDepInfo)}

    def start_step(
        self,
        *,
        flux_value: float,
        flux_idx: int,
        cur_m: float,
        m_ratio: float,
    ) -> None:
        self.current = FluxDepInfo()
        self.update(
            flux_value=flux_value,
            flux_idx=flux_idx,
            cur_m=cur_m,
            m_ratio=m_ratio,
        )

    def update(self, **values: Any) -> None:
        for name, value in values.items():
            if name not in self._field_names:
                raise AttributeError(f"Unknown FluxDepInfo field: {name}")
            copied = deepcopy(value)
            setattr(self.current, name, copied)
            if getattr(self.first, name) is None:
                setattr(self.first, name, deepcopy(value))
            setattr(self.last, name, deepcopy(value))

    def last_or(self, name: str, fallback: Any, *, task_name: str | None = None) -> Any:
        if name not in self._field_names:
            raise AttributeError(f"Unknown FluxDepInfo field: {name}")
        value = getattr(self.last, name)
        return fallback if value is None else value

    def require(self, name: str, *, task_name: str | None = None) -> Any:
        if name not in self._field_names:
            raise AttributeError(f"Unknown FluxDepInfo field: {name}")
        value = getattr(self.current, name)
        if value is None:
            owner = "FluxDepInfo" if task_name is None else f"{task_name} FluxDepInfo"
            raise ValueError(f"{owner}.{name} is required but has not been set")
        return value

    @property
    def flux_value(self) -> float:
        return float(self.require("flux_value"))

    @property
    def flux_idx(self) -> int:
        return int(self.require("flux_idx"))

    @property
    def predict_freq(self) -> float:
        return float(self.require("predict_freq"))

    @property
    def best_ro_freq(self) -> float | None:
        return self.current.best_ro_freq

    @property
    def best_ro_gain(self) -> float | None:
        return self.current.best_ro_gain


@dataclass(slots=True)
class FluxDepEnv:
    soc: object
    soccfg: object
    ml: ModuleLibrary
    flux_values: NDArray[np.float64]
    predictor: FluxoniumPredictor
    info: FluxDepInfoTracker
