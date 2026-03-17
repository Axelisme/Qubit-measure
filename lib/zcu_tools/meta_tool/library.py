from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml
from typing_extensions import Any, Dict, Optional, Union, cast

from zcu_tools.device import GlobalDeviceManager
from zcu_tools.program.v2 import Module, ModuleCfg, WaveformCfg
from zcu_tools.utils import deepupdate, numpy2number

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
    def represent_dict(self, data: dict):
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


class ModuleLibrary(SyncFile):
    def __init__(
        self, cfg_path: Optional[Union[str, Path]] = None, read_only: bool = False
    ) -> None:
        self.waveforms: Dict[str, WaveformCfg] = {}
        self.modules: Dict[str, ModuleCfg] = {}
        self.read_only = read_only

        super().__init__(cfg_path)

    @auto_sync("read")
    def clone(self, dst_path: Optional[Union[str, Path]] = None) -> ModuleLibrary:
        if dst_path is not None and Path(dst_path).exists():
            raise FileExistsError(f"Destination path {dst_path} already exists")

        ml = ModuleLibrary(dst_path, read_only=False)
        ml.waveforms = deepcopy(self.waveforms)
        ml.modules = deepcopy(self.modules)
        ml.dump()

        return ml

    def check_can_write(self) -> None:
        if self.read_only:
            raise RuntimeError("ModuleLibrary is read-only")

    def _load(self, path: str) -> None:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)

        self.waveforms = cfg["waveforms"]
        self.modules = cfg["modules"]

    def _dump(self, path: str) -> None:
        self.check_can_write()

        dump_cfg = dict(waveforms=self.waveforms, modules=self.modules)
        dump_cfg = numpy2number(deepcopy(dump_cfg))

        with open(path, "w") as f:
            yaml.dump(dump_cfg, f, Dumper=ModuleDumper, sort_keys=False)

    def make_cfg(self, exp_cfg: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        exp_cfg = deepcopy(exp_cfg)
        deepupdate(exp_cfg, kwargs, behavior="force")

        # derive device configuration from global device manager
        dev_cfg = GlobalDeviceManager.get_all_info()
        deepupdate(dev_cfg, exp_cfg.get("dev", {}), behavior="force")
        exp_cfg["dev"] = dev_cfg

        modules: Dict[str, Union[str, dict]] = exp_cfg.get("modules", {})
        for name, sub_cfg in modules.items():
            if isinstance(sub_cfg, str):
                sub_cfg = self.get_module(sub_cfg)
            if "type" not in sub_cfg:
                raise ValueError(f"Top-level module {name} is missing 'type' field")

            module_cls = Module.parse(sub_cfg["type"])
            modules[name] = module_cls.auto_fill(sub_cfg, self)  # type: ignore

        exp_cfg.setdefault("relax_delay", 0.0)
        exp_cfg.setdefault("rounds", 1)
        exp_cfg.setdefault("reps", 1)

        return numpy2number(exp_cfg)

    @auto_sync("write")
    def register_waveform(self, **wav_kwargs) -> None:
        self.check_can_write()

        wav_kwargs = deepcopy(wav_kwargs)

        # filter out non-waveform attributes
        changed = False
        for name, wav_cfg in wav_kwargs.items():
            if self.waveforms.get(name) != wav_cfg:
                changed = True
            self.waveforms[name] = wav_cfg  # directly overwrite
        if changed:
            self._dirty = True

    @auto_sync("write")
    def register_module(self, **mod_kwargs) -> None:
        self.check_can_write()

        mod_kwargs = deepcopy(mod_kwargs)

        changed = False
        for name, cfg in mod_kwargs.items():
            if self.modules.get(name) != cfg:
                changed = True
            self.modules[name] = cfg

        if changed:
            self._dirty = True

    @auto_sync("read")
    def get_waveform(
        self, name: str, override_cfg: Optional[Dict[str, Any]] = None
    ) -> WaveformCfg:
        waveform = deepcopy(self.waveforms[name])
        if override_cfg is not None:
            deepupdate(cast(dict, waveform), override_cfg, behavior="force")
        return waveform  # type: ignore[return-value]

    @auto_sync("read")
    def get_module(
        self, name: str, override_cfg: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if name not in self.modules:
            raise ValueError(
                f"Module {name} not found, available modules: {list(self.modules.keys())}"
            )
        module = deepcopy(self.modules[name])
        if override_cfg is not None:
            deepupdate(cast(dict, module), override_cfg, behavior="force")
        return cast(dict, module)

    @auto_sync("write")
    def update_module(self, name: str, override_cfg: Dict[str, Any]) -> None:
        self.check_can_write()

        deepupdate(
            cast(dict, self.modules[name]), deepcopy(override_cfg), behavior="force"
        )
        self._dirty = True

    def __str__(self) -> str:
        self.sync()
        waveforms_str = ", ".join([f"{name}" for name in self.waveforms.keys()])
        modules_str = ", ".join([f"{name}" for name in self.modules.keys()])
        return f"{self.__class__.__name__}(waveforms=[{waveforms_str}], modules=[{modules_str}])"

    def __repr__(self) -> str:
        return self.__str__()
