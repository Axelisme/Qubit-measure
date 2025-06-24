from copy import deepcopy
from typing import Any, Dict, Optional

import yaml

from .tools import deepupdate, numpy2number


class ModuleLibrary:
    waveforms: Dict[str, Dict[str, Any]] = {}
    modules: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register_waveform(cls, **kwargs) -> None:
        kwargs = deepcopy(kwargs)
        for name, wav_cfg in kwargs.items():
            waveform = dict(style=wav_cfg["style"], length=wav_cfg["length"])
            if waveform["style"] == "flat_top":
                waveform["raise_pulse"] = wav_cfg["raise_pulse"]
        cls.waveforms[name] = waveform

    @classmethod
    def get_waveform(cls, name: str) -> Dict[str, Any]:
        return deepcopy(cls.waveforms[name])

    @classmethod
    def register_module(cls, **kwargs) -> None:
        cls.modules.update(deepcopy(kwargs))

    @classmethod
    def get_module(
        cls, name: str, override_cfg: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        module = deepcopy(cls.modules[name])
        if override_cfg is not None:
            deepupdate(module, override_cfg)
        return module

    @classmethod
    def update_module(cls, name: str, override_cfg: Dict[str, Any]) -> None:
        deepupdate(cls.modules[name], deepcopy(override_cfg))

    @classmethod
    def dump(cls, cfg_path: str) -> None:
        if not cfg_path.endswith(".yaml"):
            cfg_path += ".yaml"

        dump_cfg = {
            "modules": numpy2number(cls.modules),
            "waveforms": numpy2number(cls.waveforms),
        }
        with open(cfg_path, "w") as f:
            yaml.dump(dump_cfg, f)

    @classmethod
    def load(cls, cfg_path: str) -> None:
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        cls.modules = cfg["modules"]
        cls.waveforms = cfg["waveforms"]
