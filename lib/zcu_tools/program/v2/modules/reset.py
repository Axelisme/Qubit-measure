from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, ClassVar, Dict, Literal, Type, TypedDict, Union

from qick.asm_v2 import QickParam

from ..base import MyProgramV2
from .base import Module
from .pulse import Pulse, PulseCfg, check_block_mode


class NoneResetCfg(TypedDict):
    type: Literal["none"]


class PulseResetCfg(TypedDict):
    type: Literal["pulse"]
    pulse_cfg: PulseCfg


class TwoPulseResetCfg(TypedDict):
    type: Literal["two_pulse"]
    pulse1_cfg: PulseCfg
    pulse2_cfg: PulseCfg


class BathResetCfg(TypedDict):
    type: Literal["bath"]
    qubit_tone_cfg: PulseCfg
    cavity_tone_cfg: PulseCfg
    pi2_cfg: PulseCfg


ResetCfg = Union[NoneResetCfg, PulseResetCfg, TwoPulseResetCfg, BathResetCfg]


class AbsReset(Module):
    def __init__(self, name: str, cfg: ResetCfg) -> None: ...

    @classmethod
    def set_param(
        cls,
        reset_cfg: ResetCfg,
        param_name: str,
        param_value: Union[float, QickParam],
    ) -> ResetCfg:
        raise NotImplementedError(
            f"{cls.__name__} does not support set {param_name} params with {param_value}"
        )

    def total_length(self) -> Union[float, QickParam]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support total_length"
        )


class Reset(Module):
    SUPPORTED_TYPES: ClassVar[Dict[str, Type[AbsReset]]] = {}

    @classmethod
    def register_reset(
        cls, reset_type: str
    ) -> Callable[[Type[AbsReset]], Type[AbsReset]]:
        if reset_type in cls.SUPPORTED_TYPES:
            raise ValueError(
                f"Reset type {reset_type} already registered by {cls.SUPPORTED_TYPES[reset_type].__name__}"
            )

        def decorator(cls: Type[AbsReset]) -> Type[AbsReset]:
            Reset.SUPPORTED_TYPES[reset_type] = cls
            return cls

        return decorator

    @classmethod
    def get_reset_cls(cls, reset_cfg: ResetCfg) -> Type[AbsReset]:
        if reset_cfg["type"] not in cls.SUPPORTED_TYPES:
            raise ValueError(f"Unknown reset type: {reset_cfg['type']}")
        return cls.SUPPORTED_TYPES[reset_cfg["type"]]

    @classmethod
    def set_param(
        cls,
        reset_cfg: ResetCfg,
        param_name: str,
        param_value: Union[float, QickParam],
    ) -> ResetCfg:
        return cls.get_reset_cls(reset_cfg).set_param(
            reset_cfg, param_name, param_value
        )

    def __init__(self, name: str, cfg: ResetCfg) -> None:
        self.name = name
        self.cfg = deepcopy(cfg)
        self.reset = self.get_reset_cls(cfg)(name, cfg)

    def init(self, prog: MyProgramV2) -> None:
        self.reset.init(prog)

    def run(
        self, prog: MyProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        return self.reset.run(prog, t)

    def total_length(self) -> Union[float, QickParam]:
        return self.reset.total_length()


@Reset.register_reset("none")
class NoneReset(AbsReset):
    def __init__(self, name: str, cfg: NoneResetCfg) -> None:
        self.name = name

    def init(self, prog: MyProgramV2) -> None:
        pass

    def run(
        self, prog: MyProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        return t


@Reset.register_reset("pulse")
class PulseReset(AbsReset):
    def __init__(self, name: str, cfg: PulseResetCfg) -> None:
        self.name = name
        pulse_cfg = cfg["pulse_cfg"]
        self.reset_pulse = Pulse(name=f"{name}_pulse", cfg=pulse_cfg)
        check_block_mode(self.reset_pulse.name, pulse_cfg, want_block=True)

    def init(self, prog: MyProgramV2) -> None:
        self.reset_pulse.init(prog)

    def run(
        self, prog: MyProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        return self.reset_pulse.run(prog, t)

    @staticmethod
    def set_param(
        reset_cfg: PulseResetCfg, param_name: str, param_value: Union[float, QickParam]
    ) -> PulseResetCfg:
        Pulse.set_param(reset_cfg["pulse_cfg"], param_name, param_value)

        return reset_cfg


@Reset.register_reset("two_pulse")
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

    def run(
        self, prog: MyProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        t = self.reset_pulse1.run(prog, t)
        t = self.reset_pulse2.run(prog, t)
        return t

    @staticmethod
    def set_param(
        reset_cfg: TwoPulseResetCfg,
        param_name: str,
        param_value: Union[float, QickParam],
    ) -> TwoPulseResetCfg:
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

        return reset_cfg


@Reset.register_reset("bath")
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

    def run(
        self, prog: MyProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        t = self.qub_pulse.run(prog, t)
        t = self.res_pulse.run(prog, t)
        t = self.pi2_pulse.run(prog, t)
        return t

    @staticmethod
    def set_param(
        reset_cfg: BathResetCfg,
        param_name: str,
        param_value: Union[float, QickParam],
    ) -> BathResetCfg:
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

        return reset_cfg
