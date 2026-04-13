from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml
from typing_extensions import Any, Optional, TypeVar, Union, cast
from yaml.nodes import MappingNode

from zcu_tools.device import GlobalDeviceManager
from zcu_tools.program.v2 import ModuleCfg, WaveformCfg
from zcu_tools.utils import deepupdate, format_dict

from .syncfile import SyncFile, auto_sync


class ModuleDumper(yaml.SafeDumper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_indent_level = 0

    # --- 邏輯 1：縮排減少時插入空白行 ---
    def write_line_break(self, data=None):
        current_indent_level = len(self.indents)
        super().write_line_break(data)
        if current_indent_level < self.last_indent_level and current_indent_level > 0:
            super().write_line_break(data)
        self.last_indent_level = current_indent_level

    # --- 邏輯 2：將字典類型的 Value 排到最後 ---
    def represent_dict(self, data) -> MappingNode:
        data = cast(dict, data)
        # 將 dict 拆解為 (key, value)
        items = list(data.items())

        # priorities: str > int > float > bool > other > list > dict
        _type_weights = {
            "str": 0,
            "int": 1,
            "float": 2,
            "bool": 3,
            "list": 5,
            "dict": 6,
        }

        items.sort(key=lambda x: _type_weights.get(type(x[1]).__name__, 4))

        # 使用排序後的 list 重新構建 mapping 節點
        return self.represent_mapping("tag:yaml.org,2002:map", items)


# 註冊自定義的 dict 處理函數
ModuleDumper.add_representer(dict, ModuleDumper.represent_dict)

T_ModuleCfg = TypeVar("T_ModuleCfg", bound=ModuleCfg)
T_WaveformCfg = TypeVar("T_WaveformCfg", bound=WaveformCfg)

class ModuleLibrary(SyncFile):
    def __init__(
        self, cfg_path: Optional[Union[str, Path]] = None, readonly: bool = False
    ) -> None:
        self.waveforms: dict[str, WaveformCfg] = {}
        self.modules: dict[str, ModuleCfg] = {}

        super().__init__(cfg_path, readonly=readonly)

    @auto_sync("read")
    def clone(self, dst_path: Optional[Union[str, Path]] = None) -> ModuleLibrary:
        if dst_path is not None and Path(dst_path).exists():
            raise FileExistsError(f"Destination path {dst_path} already exists")

        ml = self.__class__(dst_path, readonly=False)
        ml.waveforms = deepcopy(self.waveforms)
        ml.modules = deepcopy(self.modules)
        if dst_path is not None:
            ml.dump()

        return ml

    def _load(self, path: str) -> None:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)

        for key in ["waveforms", "modules"]:
            if key not in cfg:
                raise ValueError(
                    f"{key} not found in loading module library config file"
                )
            if not isinstance(cfg[key], dict):
                raise ValueError(
                    f"{key} in loading module library config file is not a dict, "
                    f"got {type(cfg[key])}"
                )

        self.waveforms = {
            name: WaveformCfg.from_dict(wav_cfg, self)
            for name, wav_cfg in cfg["waveforms"].items()
        }
        self.modules = {
            name: ModuleCfg.from_dict(mod_cfg, self)
            for name, mod_cfg in cfg["modules"].items()
        }

    def _dump(self, path: str) -> None:
        dump_cfg = dict(waveforms=self.waveforms, modules=self.modules)
        dump_cfg = format_dict(dump_cfg)

        with open(path, "w") as f:
            yaml.dump(dump_cfg, f, Dumper=ModuleDumper, sort_keys=False)

    def make_cfg(self, exp_cfg: dict[str, Any], **kwargs) -> dict[str, Any]:
        exp_cfg = deepcopy(exp_cfg)
        deepupdate(exp_cfg, kwargs, behavior="force")

        # derive device configuration from global device manager
        dev_cfg = GlobalDeviceManager.get_all_info()
        deepupdate(dev_cfg, exp_cfg.get("dev", {}), behavior="force")
        exp_cfg["dev"] = dev_cfg

        modules: dict[str, Union[str, dict, ModuleCfg]] = exp_cfg.get("modules", {})
        for name, sub_cfg in modules.items():
            modules[name] = ModuleCfg.from_raw(sub_cfg, self)

        exp_cfg.setdefault("relax_delay", 0.0)
        exp_cfg.setdefault("rounds", 1)
        exp_cfg.setdefault("reps", 1)

        return exp_cfg

    @auto_sync("write")
    def register_waveform(
        self, **wav_kwargs: Union[dict[str, Any], WaveformCfg]
    ) -> None:
        self._check_can_write()
        wav_kwargs = deepcopy(wav_kwargs)

        # filter out non-waveform attributes
        for name, wav_cfg in wav_kwargs.items():
            if isinstance(wav_cfg, dict):
                wav_cfg = WaveformCfg.from_dict(wav_cfg, self)
            self.waveforms[name] = wav_cfg  # directly overwrite
        self._dirty = True

    @auto_sync("write")
    def register_module(self, **mod_kwargs: Union[dict[str, Any], ModuleCfg]) -> None:
        self._check_can_write()
        mod_kwargs = deepcopy(mod_kwargs)

        for name, mod_cfg in mod_kwargs.items():
            if isinstance(mod_cfg, dict):
                mod_cfg = ModuleCfg.from_dict(mod_cfg, self)
            self.modules[name] = mod_cfg  # directly overwrite

        self._dirty = True

    @auto_sync("read")
    def get_waveform(
        self,
        name: str,
        override_cfg: Optional[dict[str, Any]] = None,
        type: type[T_WaveformCfg] = WaveformCfg,
    ) -> T_WaveformCfg:
        if name not in self.waveforms:
            raise ValueError(
                f"Waveform {name} not found, available waveforms: {list(self.waveforms.keys())}"
            )
        waveform = self.waveforms[name]

        if not isinstance(waveform, type):
            raise ValueError(f"Waveform {name} is not required type {type.__name__}")

        if override_cfg is not None:
            waveform = waveform.with_updates(**override_cfg)

        return deepcopy(waveform)

    @auto_sync("read")
    def get_module(
        self,
        name: str,
        override_cfg: Optional[dict[str, Any]] = None,
        type: type[T_ModuleCfg] = ModuleCfg,
    ) -> T_ModuleCfg:
        if name not in self.modules:
            raise ValueError(
                f"Module {name} not found, available modules: {list(self.modules.keys())}"
            )
        module = self.modules[name]

        if not isinstance(module, type):
            raise ValueError(f"Module {name} is not required type {type.__name__}")

        if override_cfg is not None:
            module = module.with_updates(**override_cfg)

        return deepcopy(module)

    @auto_sync("write")
    def update_module(self, name: str, override_cfg: dict[str, Any]) -> None:
        self._check_can_write()
        self.modules[name] = self.modules[name].with_updates(**override_cfg)
        self._dirty = True

    def __str__(self) -> str:
        self.sync()
        waveforms_str = ", ".join([f"{name}" for name in self.waveforms.keys()])
        modules_str = ", ".join([f"{name}" for name in self.modules.keys()])
        return f"{self.__class__.__name__}(waveforms=[{waveforms_str}], modules=[{modules_str}])"

    def __repr__(self) -> str:
        return self.__str__()
