from typing import Any, Dict, List, Union

from qick import QickConfig
from qick.asm_v2 import Macro

from .base import MyProgramV2
from .modules import Module


class ModularProgramV2(MyProgramV2):
    """
    A class that allows custom behavior based on the provided modules.
    """

    def __init__(
        self,
        soccfg: QickConfig,
        cfg: Dict[str, Any],
        modules: List[Union[Macro, Module]],
        **kwargs,
    ) -> None:
        self.modules = modules
        super().__init__(soccfg, cfg, **kwargs)

    def _initialize(self, cfg: Dict[str, Any]) -> None:
        super()._initialize(cfg)

        for module in self.modules:
            if isinstance(module, Module):
                module.init(self)

    def _body(self, cfg: Dict[str, Any]) -> None:
        for module in self.modules:
            if isinstance(module, Macro):
                self.append_macro(module)
            else:
                assert isinstance(module, Module)
                module.run(self)


class BaseCustomProgramV2(ModularProgramV2):
    """
    A base class for custom programs to inherit.
    """

    def __init__(self, soccfg: QickConfig, cfg: Dict[str, Any], **kwargs) -> None:
        super().__init__(soccfg, cfg, modules=self.make_modules(cfg), **kwargs)

    def make_modules(self, cfg: Dict[str, Any]) -> List[Module]:
        return []
