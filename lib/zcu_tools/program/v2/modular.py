from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import override

import numpy as np
from numpy.typing import NDArray
from qick import QickConfig
from qick.asm_v2 import AsmV2, Macro

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
        sweep: list[tuple[str, SweepCfg | int]] | None = None,
        **kwargs,
    ) -> None:
        self.modules = modules
        self.sweep_dict = sweep
        self._dmem_buffer: list[int] = []
        self._protected_gen_routes: dict[int, str] = {}
        self._protected_ro_routes: dict[int, str] = {}

        logger.debug(
            "ModularProgramV2.__init__: %d modules, reps=%s, relax_delay=%s",
            len(modules),
            cfg.reps,
            cfg.relax_delay,
        )

        super().__init__(soccfg, cfg, **kwargs)

    @override
    def _initialize(self, cfg: ProgramV2Cfg) -> None:
        # add v2 sweep loops
        if self.sweep_dict is not None:
            for name, sweep in self.sweep_dict:
                count = sweep.expts if isinstance(sweep, SweepCfg) else sweep
                self.add_loop(name, count=count)

        # initialize modules
        for module in self.modules:
            module.init(self)

        logger.debug(
            "ModularProgramV2._initialize: registered %d unique pulses",
            self.pulse_registry.count,
        )

    @override
    def _body(self, cfg: ProgramV2Cfg) -> None:
        t = 0.0
        for module in self.modules:
            if logger.isEnabledFor(logging.DEBUG):
                self.debug_macro(f"{type(module).__name__}({module.name})", t)
            t = module.run(self, t)
        self.delay(t=t)

    def validate_module_execution(self) -> None:
        """Require setup/body behavior to be fully described by declared Modules.

        Semantic consumers call this before interpreting modules rather than raw
        emitted instructions. Overriding make_program, _initialize, _body or
        _cleanup can introduce ISA outside that description, so validation raises
        ValueError naming the concrete program type. make_modules overrides
        remain supported. It does
        not compile, execute, mutate, or validate individual Module capabilities;
        normal hardware acquisition does not call this method.
        """
        if (
            type(self)._body is not ModularProgramV2._body
            or type(self)._initialize is not ModularProgramV2._initialize
            or type(self).make_program is not ModularProgramV2.make_program
            or type(self)._cleanup is not ModularProgramV2._cleanup
        ):
            raise ValueError(
                f"{type(self).__name__} raw program body/setup simulation unsupported"
            )

    def protect_readout_routes(
        self, owner: str, gen_ch: int, ro_chs: Sequence[int]
    ) -> None:
        """Protect successfully declared generator/ADC channels from redeclaration.

        Call after a module's declarations succeed, including in nested module
        init. owner is its diagnostic name; gen_ch and ro_chs are explicit
        logical channel indices already declared on this program. Future
        declare_gen/readout calls for these routes raise ValueError naming the
        owner and channel. Unprotected routes retain QICK's overwrite semantics.
        Missing declarations or already protected routes raise ValueError before
        changing any protection. No hardware lookup or route inference occurs.
        """
        if gen_ch not in self.gen_chs:
            raise ValueError(f"{owner}: generator route {gen_ch} is not declared")
        if gen_ch in self._protected_gen_routes:
            raise ValueError(
                f"{owner}: generator route {gen_ch} already protected by "
                f"{self._protected_gen_routes[gen_ch]}"
            )
        for ch in ro_chs:
            if ch not in self.ro_chs:
                raise ValueError(f"{owner}: readout route {ch} is not declared")
            if ch in self._protected_ro_routes:
                raise ValueError(
                    f"{owner}: readout route {ch} already protected by "
                    f"{self._protected_ro_routes[ch]}"
                )
        self._protected_gen_routes[gen_ch] = owner
        self._protected_ro_routes.update((ch, owner) for ch in ro_chs)

    def declare_gen(self, ch: int, *args, **kwargs) -> None:
        """Delegate QICK declaration arguments unchanged unless ch is protected.

        A protected generator raises ValueError with its owner and channel;
        unprotected generators retain the inherited QICK declaration contract.
        """
        if ch in self._protected_gen_routes:
            raise ValueError(
                f"{self._protected_gen_routes[ch]}: generator route {ch} is protected"
            )
        super().declare_gen(ch, *args, **kwargs)

    def declare_readout(self, ch: int, *args, **kwargs) -> None:
        """Delegate QICK declaration arguments unchanged unless ch is protected.

        A protected ADC raises ValueError with its owner and channel;
        unprotected ADCs retain the inherited QICK declaration contract.
        """
        if ch in self._protected_ro_routes:
            raise ValueError(
                f"{self._protected_ro_routes[ch]}: readout route {ch} is protected"
            )
        super().declare_readout(ch, *args, **kwargs)

    def append_loop_after(self, name: str, *macros: Macro) -> None:
        """Append macros to QICK's exec-after hook for one declared sweep loop."""
        for idx, (loop_name, count, before, after) in enumerate(self.loops):
            if loop_name != name:
                continue
            if after is None:
                after = AsmV2()
                self.loops[idx] = (loop_name, count, before, after)
            for macro in macros:
                after.append_macro(macro)
            return
        raise ValueError(f"sweep loop {name!r} is not declared")

    def add_dmem(self, values: Sequence[int]) -> int:
        offset = len(self._dmem_buffer)
        self._dmem_buffer.extend(values)
        return offset

    # tolerated: QICK infers None from its empty base implementation; compile_datamem documents ndarray or None.
    def compile_datamem(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
    ) -> NDArray[np.int32] | None:
        if len(self._dmem_buffer) == 0:
            return None

        return np.array(self._dmem_buffer, dtype=np.int32)


class BaseCustomProgramV2(ModularProgramV2, ABC):
    """
    A base class for custom programs to inherit.
    """

    def __init__(
        self,
        soccfg: QickConfig,
        cfg: ProgramV2Cfg,
        sweep: list[tuple[str, SweepCfg | int]] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            soccfg, cfg, modules=self.make_modules(cfg), sweep=sweep, **kwargs
        )

    @abstractmethod
    def make_modules(self, cfg: ProgramV2Cfg) -> Sequence[Module]: ...
