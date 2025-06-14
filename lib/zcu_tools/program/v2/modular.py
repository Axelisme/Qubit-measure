from typing import Any, Dict, List

from myqick import QickConfig

from .base import MyProgramV2
from .modules import Module


class ModularProgramV2(MyProgramV2):
    def __init__(
        self, soccfg: QickConfig, cfg: Dict[str, Any], modules: List[Module], **kwargs
    ) -> None:
        self.modules = modules
        super().__init__(soccfg, cfg, **kwargs)

    def _initialize(self, cfg: Dict[str, Any]) -> None:
        super()._initialize(cfg)

        for module in self.modules:
            module.init(self)

    def _body(self, cfg: Dict[str, Any]) -> None:
        for module in self.modules:
            module.run(self)
