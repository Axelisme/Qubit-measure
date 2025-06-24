from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

from .default_cfg import ModuleLibrary
from .tools import deepupdate, numpy2number

NQZ_THRESHOLD = 2000  # MHz


# Function to create and configure an experiment configuration dictionary
def make_cfg(exp_cfg: Dict[str, Any], **kwargs) -> Dict[str, Any]:
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

    auto_derive(exp_cfg)

    return numpy2number(exp_cfg)


# Function to automatically derive waveform parameters based on pulse configuration
def auto_derive_waveform(pulse_cfg: Dict[str, Any]) -> None:
    """
    Automatically derive waveform parameters for a given pulse configuration.

    Args:
        pulse_cfg (Dict[str, Any]): The pulse configuration dictionary.

    Notes:
        - If the pulse style is "flat_top", a raise pulse is added with default parameters.
        - For "gauss" and "drag" styles, the sigma value is derived based on the pulse length.
        - For "drag" style, the alpha parameter is set to a default value if not provided.
    """
    # style and length are required to derive waveform
    if "style" not in pulse_cfg:
        return  # do nothing

    style: str = pulse_cfg["style"]
    length: Optional[float] = pulse_cfg.get("length")
    if style == "flat_top":
        raise_cfg: Dict[str, Any] = pulse_cfg.setdefault("raise_pulse", {})
        raise_cfg.setdefault("style", "cosine")  # default use cosine
        if length is not None:
            # default raise pulse is 10% of the total length
            # the minimum length is 15 ns
            raise_cfg.setdefault("length", 0.1 * max(length, 0.15))

        # derive raise pulse parameters
        auto_derive_waveform(raise_cfg)
    else:
        if style in ["gauss", "drag"]:
            if length is not None:
                # default sigma is 1/4 of the total length
                pulse_cfg.setdefault("sigma", length / 4)

        if style == "drag":
            pulse_cfg.setdefault("alpha", 0.5)


# Function to automatically derive pulse parameters based on pulse name and configuration
def auto_derive_pulse(
    name: str, pulse_cfg: Union[str, Dict[str, Any]]
) -> Dict[str, Any]:
    pulse_cfg = deepcopy(pulse_cfg)

    # load pulse configuration if it is a string
    if isinstance(pulse_cfg, str):
        name = pulse_cfg
        pulse_cfg = deepcopy(ModuleLibrary.get_pulse(name))
        pulse_cfg["name"] = name

    # phase
    pulse_cfg.setdefault("phase", 0.0)  # deg

    # derive waveform
    auto_derive_waveform(pulse_cfg)

    return pulse_cfg


def auto_derive_module(
    name: str, module_cfg: Union[str, Dict[str, Any]]
) -> Dict[str, Any]:
    module_cfg = deepcopy(module_cfg)

    # load module configuration if it is a string
    if isinstance(module_cfg, str):
        name = module_cfg
        module_cfg = deepcopy(ModuleLibrary.get_module(name))
        module_cfg["name"] = name

    # derive pulse in module
    for key, value in module_cfg.items():
        if is_pulse_cfg(key, value):
            module_cfg[key] = auto_derive_pulse(key, value)

    return module_cfg


def is_module_cfg(name: str, module_cfg: Any) -> bool:
    # TODO: use better method to check if it is a module configuration
    if "reset" in name or "readout" in name:
        if isinstance(module_cfg, dict):
            return "type" in module_cfg
        return isinstance(module_cfg, str)
    return False


def is_pulse_cfg(name: str, pulse_cfg: Any) -> bool:
    # TODO: use better method to check if it is a pulse configuration
    if "pulse" in name:
        if isinstance(pulse_cfg, dict):
            return "style" in pulse_cfg
        return isinstance(pulse_cfg, str)
    return False


# Function to automatically derive experiment configuration parameters
def auto_derive(exp_cfg: Dict[str, Any]) -> None:
    exp_cfg.setdefault("dev", dict(flux_dev="none", flux=0.0))
    exp_cfg.setdefault("reset", dict(type="none"))

    # derive pulse
    # TODO: better way to derive module and pulse
    for name, sub_cfg in exp_cfg.items():
        if is_module_cfg(name, sub_cfg):
            exp_cfg[name] = auto_derive_module(name, sub_cfg)
        elif is_pulse_cfg(name, sub_cfg):
            exp_cfg[name] = auto_derive_pulse(name, sub_cfg)
