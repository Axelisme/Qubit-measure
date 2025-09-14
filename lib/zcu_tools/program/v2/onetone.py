from typing import Any, Dict, List
from copy import deepcopy

from zcu_tools.library import ModuleLibrary

from .modular import BaseCustomProgramV2
from .modules import (
    Module,
    make_readout,
    make_reset,
    derive_readout_cfg,
    derive_reset_cfg,
)


class OneToneProgram(BaseCustomProgramV2):
    def make_modules(self, cfg: Dict[str, Any]) -> List[Module]:
        return [
            make_reset("reset", reset_cfg=cfg.get("reset")),
            make_readout("readout", readout_cfg=cfg["readout"]),
        ]

    @classmethod
    def derive_cfg(cls, ml: ModuleLibrary, cfg: Dict[str, Any]) -> Dict[str, Any]:
        cfg = deepcopy(cfg)

        if "reset" in cfg:
            cfg["reset"] = derive_reset_cfg(ml, cfg["reset"])
        cfg["readout"] = derive_readout_cfg(ml, cfg["readout"])

        return cfg
