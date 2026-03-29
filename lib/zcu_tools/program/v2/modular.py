from __future__ import annotations

from qick import QickConfig
from typing_extensions import Any, Mapping, NotRequired, Optional, Sequence, cast

from zcu_tools.config import config

from ..base import SweepCfg
from .base import MyProgramV2, ProgramV2Cfg
from .modules import Module
from .utils import PrintTimeStamp


class ModularProgramCfg(ProgramV2Cfg):
    modules: Mapping[str, Any]
    sweep: NotRequired[Mapping[str, SweepCfg]]


class ModularProgramV2(MyProgramV2):
    """
    A class that allows custom behavior based on the provided modules.
    """

    def __init__(
        self,
        soccfg: QickConfig,
        cfg: Mapping[str, Any],
        modules: Sequence[Module],
        sweep: Optional[list[tuple[str, SweepCfg]]] = None,
        **kwargs,
    ) -> None:
        _cfg = cast(ModularProgramCfg, cfg)

        self.modules = modules
        self.sweep = sweep

        super().__init__(soccfg, _cfg, **kwargs)

    def _initialize(self, cfg: ModularProgramCfg) -> None:
        super()._initialize(cfg)

        # add v2 sweep loops
        if self.sweep is not None:
            for name, sweep in self.sweep:
                self.add_loop(name, count=sweep["expts"])

        # initialize modules
        for module in self.modules:
            module.init(self)

    def _body(self, cfg: ModularProgramCfg) -> None:

        t = 0.0
        for module in self.modules:
            if config.DEBUG_MODE:
                self.append_macro(
                    PrintTimeStamp(f"{module.__class__.__name__}({module.name})", t)
                )
            t = module.run(self, t)

        self.delay(t=t)


class BaseCustomProgramV2(ModularProgramV2):
    """
    A base class for custom programs to inherit.
    """

    def __init__(
        self,
        soccfg: QickConfig,
        cfg: Mapping[str, Any],
        sweep: Optional[list[tuple[str, SweepCfg]]] = None,
        **kwargs,
    ) -> None:
        _cfg = cast(ModularProgramCfg, cfg)

        super().__init__(
            soccfg, _cfg, modules=self.make_modules(_cfg), sweep=sweep, **kwargs
        )

    def make_modules(self, cfg: ModularProgramCfg) -> Sequence[Module]:
        return []
