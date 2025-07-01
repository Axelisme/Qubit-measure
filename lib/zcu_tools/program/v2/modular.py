from typing import Any, Dict, List

from qick import QickConfig

from .base import MyProgramV2
from .modules import Module


class ModularProgramV2(MyProgramV2):
    def __init__(self, soccfg: QickConfig, cfg: Dict[str, Any], **kwargs) -> None:
        self.modules = self.make_modules(cfg)
        super().__init__(soccfg, cfg, **kwargs)

    def make_modules(self, cfg: Dict[str, Any]) -> List[Module]:
        return []

    def _initialize(self, cfg: Dict[str, Any]) -> None:
        super()._initialize(cfg)

        for module in self.modules:
            module.init(self)

    def _body(self, cfg: Dict[str, Any]) -> None:
        for module in self.modules:
            module.run(self)
