from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type, TypeVar

from ..base import MyProgramV2
from .base import Module
from .pulse import Pulse, check_block_mode

RESET_REGISTRY: Dict[str, Type[AbsReset]] = {}

T_Reset = TypeVar("T_Reset", bound="AbsReset")


def register_reset(reset_type: str) -> Callable[[T_Reset], T_Reset]:
    def decorator(cls: T_Reset) -> T_Reset:
        RESET_REGISTRY[reset_type] = cls
        return cls

    return decorator


class AbsReset(Module):
    pass


def make_reset(name: str, reset_cfg: Optional[Dict[str, Any]]) -> AbsReset:
    if reset_cfg is None:
        return NoneReset(name, None)

    reset_type = reset_cfg.get("type", "none")
    reset_cls = RESET_REGISTRY.get(reset_type)

    if reset_cls is None:
        raise ValueError(f"Unknown reset type: {reset_type}")

    return reset_cls(name, cfg=reset_cfg)


def set_reset_cfg(
    reset_cfg: Optional[Dict[str, Any]], param_name: str, param_value: float
) -> None:
    if reset_cfg is None:
        return

    reset_type = reset_cfg["type"]
    if reset_type not in RESET_REGISTRY:
        raise ValueError(f"Unknown reset type: {reset_type}")

    reset_cls = RESET_REGISTRY[reset_type]
    if hasattr(reset_cls, "set_param"):
        reset_cls.set_param(reset_cfg, param_name, param_value)


@register_reset("none")
class NoneReset(AbsReset):
    def __init__(self, name: str, cfg: Optional[Dict[str, Any]]) -> None:
        self.name = name

    def init(self, prog: MyProgramV2) -> None:
        pass

    def run(self, prog: MyProgramV2, t: float = 0.0) -> float:
        return t


@register_reset("pulse")
class PulseReset(AbsReset):
    def __init__(self, name: str, cfg: Dict[str, Any]) -> None:
        self.name = name
        pulse_cfg = cfg["pulse_cfg"]
        self.reset_pulse = Pulse(name=f"{name}_pulse", cfg=pulse_cfg)
        check_block_mode(self.reset_pulse.name, pulse_cfg, want_block=True)

    def init(self, prog: MyProgramV2) -> None:
        self.reset_pulse.init(prog)

    def run(self, prog: MyProgramV2, t: float = 0.0) -> float:
        return self.reset_pulse.run(prog, t)

    @staticmethod
    def set_param(
        reset_cfg: Dict[str, Any], param_name: str, param_value: float
    ) -> None:
        Pulse.set_param(reset_cfg["pulse_cfg"], param_name, param_value)


@register_reset("two_pulse")
class TwoPulseReset(AbsReset):
    def __init__(self, name: str, cfg: Dict[str, Any]) -> None:
        self.name = name
        pulse1_cfg = cfg["pulse1_cfg"]
        pulse2_cfg = cfg["pulse2_cfg"]

        self.reset_pulse1 = Pulse(name=f"{name}_pulse1", cfg=pulse1_cfg)
        self.reset_pulse2 = Pulse(name=f"{name}_pulse2", cfg=pulse2_cfg)

        check_block_mode(self.reset_pulse1.name, pulse1_cfg, want_block=False)
        check_block_mode(self.reset_pulse2.name, pulse2_cfg, want_block=True)

    def init(self, prog: MyProgramV2) -> None:
        self.reset_pulse1.init(prog)
        self.reset_pulse2.init(prog)

    def run(self, prog: MyProgramV2, t: float = 0.0) -> float:
        t = self.reset_pulse1.run(prog, t)
        t = self.reset_pulse2.run(prog, t)
        return t

    @staticmethod
    def set_param(
        reset_cfg: Dict[str, Any], param_name: str, param_value: float
    ) -> None:
        if param_name == "on/off":
            Pulse.set_param(reset_cfg["pulse1_cfg"], "on/off", param_value)
            Pulse.set_param(reset_cfg["pulse2_cfg"], "on/off", param_value)
        elif param_name in ["gain1", "freq1"]:
            Pulse.set_param(
                reset_cfg["pulse1_cfg"], param_name.replace("1", ""), param_value
            )
        elif param_name in ["gain2", "freq2"]:
            Pulse.set_param(
                reset_cfg["pulse2_cfg"], param_name.replace("2", ""), param_value
            )
        elif param_name == "length":
            Pulse.set_param(reset_cfg["pulse1_cfg"], "length", param_value)
            Pulse.set_param(reset_cfg["pulse2_cfg"], "length", param_value)
        else:
            raise ValueError(f"Unknown parameter: {param_name}")


@register_reset("bath")
class BathReset(AbsReset):
    def __init__(self, name: str, cfg: Dict[str, Any]) -> None:
        self.name = name
        qubit_tone_cfg = cfg["qubit_tone_cfg"]
        cavity_tone_cfg = cfg["cavity_tone_cfg"]
        pi2_cfg = cfg["pi2_cfg"]

        self.qub_pulse = Pulse(name=f"{name}_qub_pulse", cfg=qubit_tone_cfg)
        self.res_pulse = Pulse(name=f"{name}_res_pulse", cfg=cavity_tone_cfg)
        self.pi2_pulse = Pulse(name=f"{name}_pi2_pulse", cfg=pi2_cfg)

        check_block_mode(self.qub_pulse.name, qubit_tone_cfg, want_block=False)
        check_block_mode(self.res_pulse.name, cavity_tone_cfg, want_block=True)
        check_block_mode(self.pi2_pulse.name, pi2_cfg, want_block=True)

    def init(self, prog: MyProgramV2) -> None:
        self.qub_pulse.init(prog)
        self.res_pulse.init(prog)
        self.pi2_pulse.init(prog)

    def run(self, prog: MyProgramV2, t: float = 0.0) -> float:
        t = self.qub_pulse.run(prog, t)
        t = self.res_pulse.run(prog, t)
        t = self.pi2_pulse.run(prog, t)
        return t

    @staticmethod
    def set_param(
        reset_cfg: Dict[str, Any], param_name: str, param_value: float
    ) -> None:
        if param_name == "on/off":
            Pulse.set_param(reset_cfg["qubit_tone_cfg"], "on/off", param_value)
            Pulse.set_param(reset_cfg["cavity_tone_cfg"], "on/off", param_value)
            Pulse.set_param(reset_cfg["pi2_cfg"], "on/off", param_value)
        elif param_name in ["qub_gain", "qub_freq"]:
            Pulse.set_param(
                reset_cfg["qubit_tone_cfg"], param_name.replace("qub_", ""), param_value
            )
        elif param_name in ["res_gain", "res_freq"]:
            Pulse.set_param(
                reset_cfg["cavity_tone_cfg"],
                param_name.replace("res_", ""),
                param_value,
            )
        elif param_name == "length":
            Pulse.set_param(reset_cfg["qubit_tone_cfg"], "length", param_value)
            Pulse.set_param(reset_cfg["cavity_tone_cfg"], "length", param_value)
        elif param_name == "pi2_phase":
            Pulse.set_param(reset_cfg["pi2_cfg"], "phase", param_value)
        else:
            raise ValueError(f"Unknown parameter: {param_name}")
