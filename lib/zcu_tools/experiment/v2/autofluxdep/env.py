from __future__ import annotations

from collections import UserDict
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from zcu_tools.meta_tool import ModuleLibrary
from zcu_tools.simulate.fluxonium import FluxoniumPredictor


@dataclass(slots=True)
class FluxDepDeps:
    soc: object
    soccfg: object
    ml: ModuleLibrary


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


@dataclass(slots=True)
class FluxDepEnv:
    soc: object
    soccfg: object
    ml: ModuleLibrary
    flux_values: NDArray[np.float64]
    predictor: FluxoniumPredictor
    info: FluxDepInfoDict
