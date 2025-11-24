from __future__ import annotations

import os
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    Literal,
    MutableMapping,
    Optional,
    TextIO,
    TypeVar,
    Union,
)

import yaml
from typing_extensions import ParamSpec

from zcu_tools.device import GlobalDeviceManager
from zcu_tools.utils import deepupdate, numpy2number

try:  # pragma: no cover - platform specific
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore[attr-defined]

try:  # pragma: no cover - platform specific
    import msvcrt
except ImportError:  # pragma: no cover - POSIX fallback
    msvcrt = None  # type: ignore[attr-defined]


def _acquire_file_lock(handle: TextIO) -> None:
    if fcntl is not None:  # POSIX
        fcntl.flock(handle, fcntl.LOCK_EX)  # type: ignore[attr-defined]
        return

    if msvcrt is not None:  # Windows
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write("\0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)  # type: ignore[attr-defined]
        return

    raise RuntimeError("File locking not supported on this platform.")


def _release_file_lock(handle: TextIO) -> None:
    if fcntl is not None:  # POSIX
        fcntl.flock(handle, fcntl.LOCK_UN)  # type: ignore[attr-defined]
        return

    if msvcrt is not None:  # Windows
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
        return

    raise RuntimeError("File locking not supported on this platform.")


def is_module_cfg(name: str, module_cfg: Any) -> bool:
    # TODO: use better method to check if it is a module configuration

    # common modules
    if name in ["reset", "tested_reset", "readout"]:
        return True

    # inside common modules
    if "_cfg" in name:
        return True

    # pulse modules
    if "_pulse" in name:
        return True
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

    # if it also a pulse cfg, exclude raise_waveform in flat_top
    if "waveform" in module_cfg and name != "raise_waveform":
        module_cfg.setdefault("phase", 0.0)
        module_cfg.setdefault("pre_delay", 0.0)
        module_cfg.setdefault("post_delay", 0.0)
        module_cfg.setdefault("block_mode", True)

    # derive pulse in module
    for key, value in module_cfg.items():
        if is_module_cfg(key, value):
            module_cfg[key] = auto_derive_module(ml, key, value)

    return module_cfg


P = ParamSpec("P")
T = TypeVar("T")


def auto_sync(
    time: Literal["after", "before"],
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            assert isinstance(args[0], ModuleLibrary)

            if time == "before":
                args[0].sync()

            result = func(*args, **kwargs)

            if time == "after":
                args[0].sync()

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
            os.makedirs(self.cfg_path.parent, exist_ok=True)
            self._lock_path = Path(str(self.cfg_path) + ".lock")
        else:
            self.cfg_path = None
            self._lock_path = None
        self.modify_time = 0

        self.waveforms: Dict[str, Dict[str, Any]] = {}
        self.modules: Dict[str, Dict[str, Any]] = {}
        self._dirty = False

        self.sync()

    def clone(self) -> ModuleLibrary:
        ml = ModuleLibrary()
        ml.waveforms = deepcopy(self.waveforms)
        ml.modules = deepcopy(self.modules)

        return ml

    def make_cfg(
        self, exp_cfg: MutableMapping[str, Any], **kwargs
    ) -> MutableMapping[str, Any]:
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
        assert self.cfg_path is not None

        with open(str(self.cfg_path), "r") as f:
            cfg = yaml.safe_load(f)
        self.modify_time = self.cfg_path.stat().st_mtime_ns

        deepupdate(self.waveforms, cfg["waveforms"], behavior="force")
        deepupdate(self.modules, cfg["modules"], behavior="force")
        self._dirty = False

    def dump(self) -> None:
        assert self.cfg_path is not None

        dump_cfg = dict(waveforms=self.waveforms, modules=self.modules)
        dump_cfg = numpy2number(deepcopy(dump_cfg))

        with open(str(self.cfg_path), "w") as f:
            yaml.dump(dump_cfg, f)
        self.modify_time = self.cfg_path.stat().st_mtime_ns
        self._dirty = False

    def sync(self) -> None:
        if self.cfg_path is None:
            return  # do nothing

        with self._file_lock():
            if self.cfg_path.exists():
                mtime = self.cfg_path.stat().st_mtime_ns
                if mtime > self.modify_time:
                    self.load()

            if self._dirty:
                self.dump()

    @contextmanager
    def _file_lock(self) -> Iterator[None]:
        if self._lock_path is None:
            yield
            return

        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._lock_path, "a+") as handle:
            _acquire_file_lock(handle)
            try:
                yield
            finally:
                _release_file_lock(handle)

    @auto_sync("after")
    def register_waveform(self, **wav_kwargs) -> None:
        wav_kwargs = deepcopy(wav_kwargs)

        # filter out non-waveform attributes
        changed = False
        for name, wav_cfg in wav_kwargs.items():
            if self.waveforms.get(name) != wav_cfg:
                changed = True
            self.waveforms[name] = wav_cfg  # directly overwrite
        if changed:
            self._dirty = True

    @auto_sync("after")
    def register_module(self, **mod_kwargs) -> None:
        mod_kwargs = deepcopy(mod_kwargs)

        changed = False
        for name, cfg in mod_kwargs.items():
            if self.modules.get(name) != cfg:
                changed = True
            self.modules[name] = cfg

        if changed:
            self._dirty = True

    @auto_sync("before")
    def get_waveform(
        self, name: str, override_cfg: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        waveform = deepcopy(self.waveforms[name])
        if override_cfg is not None:
            deepupdate(waveform, override_cfg, behavior="force")
        return waveform

    @auto_sync("before")
    def get_module(
        self, name: str, override_cfg: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        module = deepcopy(self.modules[name])
        if override_cfg is not None:
            deepupdate(module, override_cfg, behavior="force")
        return module

    @auto_sync("after")
    def update_module(self, name: str, override_cfg: Dict[str, Any]) -> None:
        deepupdate(self.modules[name], deepcopy(override_cfg), behavior="force")
        self._dirty = True
