from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, Any, TypeVar

from zcu_tools.device import DeviceInfo, GlobalDeviceManager
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import format_sweep1D, get_single_sweep_name
from zcu_tools.program.v2 import ModuleCfgFactory
from zcu_tools.utils import deepupdate

if TYPE_CHECKING:
    from zcu_tools.meta_tool import ModuleLibrary


T_ExpCfg = TypeVar("T_ExpCfg", bound=ExpCfgModel)


def assemble_experiment_cfg(
    raw_cfg: Mapping[str, Any],
    cfg_model: type[T_ExpCfg],
    *,
    ml: ModuleLibrary,
    device_snapshot: Mapping[str, DeviceInfo],
    overrides: Mapping[str, Any] | None = None,
) -> T_ExpCfg:
    exp_cfg = deepcopy(dict(raw_cfg))
    if overrides is not None:
        deepupdate(exp_cfg, dict(overrides), behavior="force")

    dev_cfg = dict(device_snapshot)
    for name, patch in _device_patches(exp_cfg).items():
        if name not in dev_cfg:
            raise KeyError(f"Device {name} not found in device snapshot.")
        if not isinstance(patch, Mapping):
            raise TypeError(f"Device patch for {name!r} must be a mapping.")
        dev_cfg[name] = dev_cfg[name].with_updates(**dict(patch))
    exp_cfg["dev"] = dev_cfg

    if (modules := exp_cfg.get("modules")) is not None:
        if not isinstance(modules, dict):
            raise TypeError("Experiment config field 'modules' must be a dict.")
        for name, sub_cfg in modules.items():
            modules[name] = ModuleCfgFactory.from_raw(sub_cfg, ml=ml)

    if (sweep_cfg := exp_cfg.get("sweep")) is not None:
        if (sweep_name := get_single_sweep_name(cfg_model)) is not None:
            exp_cfg["sweep"] = format_sweep1D(sweep_cfg, sweep_name)

    try:
        return cfg_model.model_validate(exp_cfg)
    except Exception as e:
        raise ValueError(
            f"Error validating experiment config with {cfg_model.__name__}:\n"
            f"exp_cfg: {exp_cfg}\n"
            f"error: {e}"
        ) from e


def make_cfg(
    raw_cfg: Mapping[str, Any],
    cfg_model: type[T_ExpCfg],
    *,
    ml: ModuleLibrary,
    overrides: Mapping[str, Any] | None = None,
    device_snapshot: Mapping[str, DeviceInfo] | None = None,
) -> T_ExpCfg:
    if device_snapshot is None:
        device_snapshot = GlobalDeviceManager.get_all_info()
    return assemble_experiment_cfg(
        raw_cfg,
        cfg_model,
        ml=ml,
        device_snapshot=device_snapshot,
        overrides=overrides,
    )


def _device_patches(exp_cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    patches = exp_cfg.get("dev", {})
    if patches is None:
        return {}
    if not isinstance(patches, Mapping):
        raise TypeError("Experiment config field 'dev' must be a mapping.")
    return patches
