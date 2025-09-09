from __future__ import annotations

from copy import deepcopy
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, TypeVar, Union

import yaml
from typing_extensions import ParamSpec

from zcu_tools.device import GlobalDeviceManager
from zcu_tools.utils import deepupdate, numpy2number


def is_module_cfg(name: str, module_cfg: Any) -> bool:
    # TODO: use better method to check if it is a module configuration
    if "reset" in name or "readout" in name or "pulse" in name:
        if isinstance(module_cfg, dict):
            return "type" in module_cfg or "style" in module_cfg
        return isinstance(module_cfg, str)
    return False


def auto_derive_module(
    ml: ModuleLibrary, name: str, module_cfg: Union[str, Dict[str, Any]]
) -> Dict[str, Any]:
    module_cfg = deepcopy(module_cfg)

    # load module configuration if it is a string
    if isinstance(module_cfg, str):
        name = module_cfg
        module_cfg = deepcopy(ml.get_module(name))
    module_cfg["name"] = name

    # if it also a pulse cfg, exclude raise_pulse in flat_top
    if "style" in module_cfg and name != "raise_pulse":
        module_cfg.setdefault("phase", 0.0)
        module_cfg.setdefault("t", "auto")
        module_cfg.setdefault("post_delay", 0.0)

    # derive pulse in module
    for key, value in module_cfg.items():
        if is_module_cfg(key, value):
            module_cfg[key] = auto_derive_module(ml, key, value)

    return module_cfg


def _sync(time: Literal["after", "before"]) -> Callable:
    P = ParamSpec("P")
    T = TypeVar("T")

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(self: ModuleLibrary, *args, **kwargs) -> T:
            if time == "before":
                self.sync()

            result = func(self, *args, **kwargs)

            if time == "after":
                self.sync()
            return result

        return wrapper

    return decorator


class ModuleLibrary:
    """
    Module library is a class that contains the waveforms and modules of the experiment.
    It is used to automatically derive the configuration of the experiment.
    """

    def __init__(self, cfg_path: Optional[str] = None) -> None:
        if cfg_path is not None:
            self.cfg_path = Path(cfg_path)
        else:
            self.cfg_path = None
        self.modify_time = 0

        self.waveforms: Dict[str, Dict[str, Any]] = {}
        self.modules: Dict[str, Dict[str, Any]] = {}

        self.sync()

    def make_cfg(self, exp_cfg: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Create a deep copy of the experiment configuration, update it with additional parameters,
        and automatically derive missing configuration values.

        Args:
            exp_cfg (Dict[str, Any]): The base experiment configuration.
            **kwargs: Additional parameters to update the configuration.

        Returns:
            Dict[str, Any]: The updated and finalized experiment configuration.
        """

        exp_cfg = deepcopy(exp_cfg)
        deepupdate(exp_cfg, kwargs, behavior="force")

        # derive device configuration from global device manager
        dev_cfg = GlobalDeviceManager.get_all_info()
        deepupdate(dev_cfg, exp_cfg.get("dev", {}), behavior="force")
        exp_cfg["dev"] = dev_cfg

        for name, sub_cfg in exp_cfg.items():
            if is_module_cfg(name, sub_cfg):
                exp_cfg[name] = auto_derive_module(self, name, sub_cfg)

        return numpy2number(exp_cfg)

    def load(self) -> None:
        with open(self.cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.modify_time = self.cfg_path.stat().st_mtime_ns

        deepupdate(self.waveforms, cfg["waveforms"], behavior="force")
        deepupdate(self.modules, cfg["modules"], behavior="force")

    def dump(self) -> None:
        dump_cfg = dict(waveforms=self.waveforms, modules=self.modules)
        dump_cfg = numpy2number(deepcopy(dump_cfg))

        with open(self.cfg_path, "w") as f:
            yaml.dump(dump_cfg, f)
        self.modify_time = self.cfg_path.stat().st_mtime_ns

    def sync(self) -> None:
        if self.cfg_path is None:
            return  # do nothing

        if self.cfg_path.exists():
            mtime = self.cfg_path.stat().st_mtime_ns
            if mtime > self.modify_time:
                self.load()

        self.dump()

    @_sync("after")
    def register_waveform(self, **wav_kwargs) -> None:
        wav_kwargs = deepcopy(wav_kwargs)

        # filter out non-waveform attributes
        for name, wav_cfg in wav_kwargs.items():
            waveform = dict(style=wav_cfg["style"], length=wav_cfg["length"])
            if waveform["style"] == "flat_top":
                waveform["raise_pulse"] = wav_cfg["raise_pulse"]
            elif waveform["style"] in ["gauss", "drag"]:
                waveform["sigma"] = wav_cfg["sigma"]

            if waveform["style"] == "drag":
                waveform["delta"] = wav_cfg["delta"]
                waveform["alpha"] = wav_cfg["alpha"]

            self.waveforms[name] = waveform  # directly overwrite

    @_sync("after")
    def register_module(self, **mod_kwargs) -> None:
        self.modules.update(deepcopy(mod_kwargs))

    @_sync("before")
    def get_waveform(self, name: str) -> Dict[str, Any]:
        return deepcopy(self.waveforms[name])

    @_sync("before")
    def get_module(
        self, name: str, override_cfg: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        module = deepcopy(self.modules[name])
        if override_cfg is not None:
            deepupdate(module, override_cfg, behavior="force")
        return module

    @_sync("after")
    def update_module(self, name: str, override_cfg: Dict[str, Any]) -> None:
        deepupdate(self.modules[name], deepcopy(override_cfg), behavior="force")
