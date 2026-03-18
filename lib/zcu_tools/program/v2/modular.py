from __future__ import annotations

from qick import QickConfig
from typing_extensions import Any, Mapping, NotRequired, Sequence, cast

from ..base import SweepCfg
from .base import MyProgramV2, ProgramV2Cfg
from .modules import Module


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
        **kwargs,
    ) -> None:
        _cfg = cast(ModularProgramCfg, cfg)

        self.modules = modules

        super().__init__(soccfg, _cfg, **kwargs)

    def _initialize(self, cfg: ModularProgramCfg) -> None:
        super()._initialize(cfg)

        # add v2 sweep loops
        for name, sweep in cfg.get("sweep", {}).items():
            self.add_loop(name, count=sweep["expts"])

        # initialize modules
        for module in self.modules:
            module.init(self)

    def _body(self, cfg: ModularProgramCfg) -> None:
        from .utils import PrintTimeStamp

        t = 0.0
        for module in self.modules:
            t = module.run(self, t)
            # self.append_macro(PrintTimeStamp(module.name, gen_chs=[0], ro_chs=[0]))
        self.delay(t=t + 0.03)


class BaseCustomProgramV2(ModularProgramV2):
    """
    A base class for custom programs to inherit.
    """

    def __init__(self, soccfg: QickConfig, cfg: Mapping[str, Any], **kwargs) -> None:
        _cfg = cast(ModularProgramCfg, cfg)

        super().__init__(soccfg, _cfg, modules=self.make_modules(_cfg), **kwargs)

    def make_modules(self, cfg: ModularProgramCfg) -> Sequence[Module]:
        return []
