from typing import Any, Dict, List
from copy import deepcopy

from zcu_tools.library import ModuleLibrary
from .modular import BaseCustomProgramV2
from .modules import (
    Module,
    Pulse,
    make_readout,
    make_reset,
    derive_readout_cfg,
    derive_reset_cfg,
)


class TwoToneProgram(BaseCustomProgramV2):
    def make_modules(self, cfg: Dict[str, Any]) -> List[Module]:
        return [
            make_reset("reset", reset_cfg=cfg.get("reset")),
            Pulse(name="init_pulse", cfg=cfg.get("init_pulse")),
            Pulse(name="qubit_pulse", cfg=cfg["qub_pulse"]),
            make_readout("readout", readout_cfg=cfg["readout"]),
        ]

    @classmethod
    def derive_cfg(cls, ml: ModuleLibrary, cfg: Dict[str, Any]) -> Dict[str, Any]:
        cfg = deepcopy(cfg)

        if "reset" in cfg:
            cfg["reset"] = derive_reset_cfg(ml, cfg["reset"])
        if "init_pulse" in cfg:
            cfg["init_pulse"] = Pulse.derive_cfg(ml, cfg["init_pulse"])
        cfg["qub_pulse"] = Pulse.derive_cfg(ml, cfg["qub_pulse"])
        cfg["readout"] = derive_readout_cfg(ml, cfg["readout"])

        return cfg
