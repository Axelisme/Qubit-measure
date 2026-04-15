from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from qick import QickConfig
from typing_extensions import Any, Mapping, Optional, Sequence, Union, cast

from ..base import SweepCfg
from .base import MyProgramV2, ProgramV2Cfg
from .modules import Module, ModuleCfg
from .utils import PrintTimeStamp

logger = logging.getLogger(__name__)


class ModularProgramCfg(ProgramV2Cfg):
    modules: Mapping[str, ModuleCfg]


class ModularProgramV2(MyProgramV2):
    """
    A class that allows custom behavior based on the provided modules.
    """

    def __init__(
        self,
        soccfg: QickConfig,
        cfg: Mapping[str, Any],
        modules: Sequence[Module],
        sweep: Optional[list[tuple[str, Union[SweepCfg, int]]]] = None,
        **kwargs,
    ) -> None:
        _cfg = cast(ModularProgramCfg, cfg)

        self.modules = modules
        self.sweep_dict = sweep
        self._dmem_buffer = []

        logger.debug(
            "ModularProgramV2.__init__: %d modules, reps=%s, relax_delay=%s",
            len(modules),
            _cfg["reps"],
            _cfg["relax_delay"],
        )

        super().__init__(soccfg, _cfg, **kwargs)

    def _initialize(self, cfg: ModularProgramCfg) -> None:
        super()._initialize(cfg)

        # add v2 sweep loops
        if self.sweep_dict is not None:
            for name, sweep in self.sweep_dict:
                if isinstance(sweep, dict):
                    self.add_loop(name, count=sweep["expts"])
                else:
                    self.add_loop(name, count=sweep)

        # initialize modules
        for module in self.modules:
            module.init(self)

        logger.debug(
            "ModularProgramV2._initialize: registered %d unique pulses, %d waveforms",
            len(self.pulse_registry._pulses),
            len(self.waves),
        )

    def _body(self, cfg: ModularProgramCfg) -> None:

        t = 0.0
        for module in self.modules:
            if logger.isEnabledFor(logging.DEBUG):
                self.debug_macro(f"{type(module).__name__}({module.name})", t)
            t = module.run(self, t)

        self.delay(t=t)

    def add_dmem(self, values: Sequence[int]) -> int:
        offset = len(self._dmem_buffer)
        self._dmem_buffer.extend(values)
        return offset

    def compile_datamem(self) -> Optional[NDArray[np.int32]]:
        if len(self._dmem_buffer) == 0:
            return None

        return np.array(self._dmem_buffer, dtype=np.int32)


class BaseCustomProgramV2(ModularProgramV2):
    """
    A base class for custom programs to inherit.
    """

    def __init__(
        self,
        soccfg: QickConfig,
        cfg: Mapping[str, Any],
        sweep: Optional[list[tuple[str, Union[SweepCfg, int]]]] = None,
        **kwargs,
    ) -> None:
        _cfg = cast(ModularProgramCfg, cfg)

        super().__init__(
            soccfg, _cfg, modules=self.make_modules(_cfg), sweep=sweep, **kwargs
        )

    def make_modules(self, cfg: ModularProgramCfg) -> Sequence[Module]:
        return []
