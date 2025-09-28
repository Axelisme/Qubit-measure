from typing import Any, Dict, List

from qick import QickConfig

from .base import MyProgramV2
from .modules import Module


class ModularProgramV2(MyProgramV2):
    """
    A class that allows custom behavior based on the provided modules.
    """

    def __init__(
        self, soccfg: QickConfig, cfg: Dict[str, Any], modules: List[Module], **kwargs
    ) -> None:
        self.modules = modules
        super().__init__(soccfg, cfg, **kwargs)

    def _initialize(self, cfg: Dict[str, Any]) -> None:
        super()._initialize(cfg)

        # add v2 sweep loops
        for name, sweep in cfg.get("sweep", {}).items():
            self.add_loop(name, count=sweep["expts"])

        # initialize modules
        for module in self.modules:
            module.init(self)

    def _body(self, cfg: Dict[str, Any]) -> None:
        t = 0.0
        for module in self.modules:
            t = module.run(self, t)

        # a slight exteral delay to avoid error in delay_auto()
        self.delay(t=t + 0.03)


class BaseCustomProgramV2(ModularProgramV2):
    """
    A base class for custom programs to inherit.
    """

    def __init__(self, soccfg: QickConfig, cfg: Dict[str, Any], **kwargs) -> None:
        super().__init__(soccfg, cfg, modules=self.make_modules(cfg), **kwargs)

    def make_modules(self, cfg: Dict[str, Any]) -> List[Module]:
        return []
