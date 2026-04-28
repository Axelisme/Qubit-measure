from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

import numpy as np
from numpy.typing import NDArray
from qick import QickConfig
from typing_extensions import Optional, Sequence, Union

from .base import MyProgramV2, ProgramV2Cfg
from .modules import Module
from .sweep import SweepCfg

logger = logging.getLogger(__name__)


class ModularProgramV2(MyProgramV2):
    """
    A class that allows custom behavior based on the provided modules.
    """

    def __init__(
        self,
        soccfg: QickConfig,
        cfg: ProgramV2Cfg,
        modules: Sequence[Module],
        sweep: Optional[list[tuple[str, Union[SweepCfg, int]]]] = None,
        **kwargs,
    ) -> None:
        self.modules = modules
        self.sweep_dict = sweep
        self._dmem_buffer = []
        self._temp_regs: list[str] = []
        self._temp_reg_scope_stack: list[int] = []

        logger.debug(
            "ModularProgramV2.__init__: %d modules, reps=%s, relax_delay=%s",
            len(modules),
            cfg.reps,
            cfg.relax_delay,
        )

        super().__init__(soccfg, cfg, **kwargs)

    def _initialize(self, cfg: ProgramV2Cfg) -> None:
        # add v2 sweep loops
        if self.sweep_dict is not None:
            for name, sweep in self.sweep_dict:
                if isinstance(sweep, SweepCfg):
                    self.add_loop(name, count=sweep.expts)
                else:
                    self.add_loop(name, count=sweep)

        # initialize modules
        for module in self.modules:
            module.init(self)

        logger.debug(
            "ModularProgramV2._initialize: registered %d unique pulses",
            len(self.pulse_registry._pulses),
        )

    def _body(self, cfg: ProgramV2Cfg) -> None:
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

    @contextmanager
    def acquire_temp_reg(self, num: int = 1) -> Iterator[list[str]]:
        """Acquire shared scratch registers for temporary calculations.

        The returned registers are shared across modules. Callers must not
        rely on temp register values persisting across module boundaries.
        """
        if num < 0:
            raise ValueError(f"num must be greater than or equal to 0, got {num}")
        elif num == 0:
            yield []
            return

        while len(self._temp_regs) < num:
            reg_name = f"temp_reg_{len(self._temp_regs)}"
            self.add_reg(reg_name, allow_reuse=True)
            self._temp_regs.append(reg_name)

        self._temp_reg_scope_stack.append(num)
        try:
            yield self._temp_regs[:num]
        finally:
            if len(self._temp_reg_scope_stack) == 0:
                raise RuntimeError("temp register scope stack is already empty")
            active_num = self._temp_reg_scope_stack.pop()
            if active_num != num:
                raise RuntimeError(
                    "temp register scope mismatch: "
                    f"expected {num} regs, got scope {active_num}"
                )

    def compile_datamem(self) -> Optional[NDArray[np.int32]]:  # type: ignore
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
        cfg: ProgramV2Cfg,
        sweep: Optional[list[tuple[str, Union[SweepCfg, int]]]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            soccfg, cfg, modules=self.make_modules(cfg), sweep=sweep, **kwargs
        )

    def make_modules(self, cfg: ProgramV2Cfg) -> Sequence[Module]:
        return []
