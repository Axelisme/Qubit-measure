from typing import Any, Dict, List

from qick import QickConfig
from qick.asm_v2 import Macro

from .base import MyProgramV2
from .modules import Module


class PrintTimeStamp(Macro):
    def __init__(self, prefix: str = "") -> None:
        self.prefix = prefix

    def expand(self, prog):
        return []

    def preprocess(self, prog):
        from pprint import pprint
        from .modules.base import param2str

        timestamps = []
        timestamps += list(prog._gen_ts)
        timestamps += list(prog._ro_ts)
        print(self.prefix)
        for i, t in enumerate(timestamps):
            print(f"\t[{i}] " + param2str(t))


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
