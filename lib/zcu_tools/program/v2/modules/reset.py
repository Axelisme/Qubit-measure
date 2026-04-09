from __future__ import annotations

from qick.asm_v2 import QickParam
from typing_extensions import (
    TYPE_CHECKING,
    Any,
    Optional,
    Type,
    TypeAlias,
    TypeVar,
    Union,
    cast,
)

from ..base import MyProgramV2
from .base import Module, ModuleCfg
from .pulse import Pulse, PulseCfg
from .util import calc_max_length

if TYPE_CHECKING:
    from zcu_tools.meta_tool import ModuleLibrary


class NoneResetCfg(ModuleCfg): ...


class PulseResetCfg(ModuleCfg):
    pulse_cfg: PulseCfg


class TwoPulseResetCfg(ModuleCfg):
    pulse1_cfg: PulseCfg
    pulse2_cfg: PulseCfg


class BathResetCfg(ModuleCfg):
    qubit_tone_cfg: PulseCfg
    cavity_tone_cfg: PulseCfg
    pi2_cfg: PulseCfg


ResetCfg: TypeAlias = Union[NoneResetCfg, PulseResetCfg, TwoPulseResetCfg, BathResetCfg]


T_ResetCfg = TypeVar("T_ResetCfg", bound=ResetCfg)


class AbsReset(Module, tag="reset"): ...


class Reset(Module):
    def __init__(self, name: str, cfg: Optional[ResetCfg]) -> None:
        self.name = name

        if cfg is None:
            cfg = {"type": "reset/none"}

        self.reset = cast(Reset, Module.parse(cfg["type"])(name, cfg))

    def init(self, prog: MyProgramV2) -> None:
        self.reset.init(prog)

    def total_length(self, prog: MyProgramV2) -> Union[float, QickParam]:
        return self.reset.total_length(prog)

    def run(
        self, prog: MyProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        return self.reset.run(prog, t)

    @staticmethod
    def set_param(
        cfg: T_ResetCfg, param_name: str, param_value: Union[float, QickParam]
    ) -> T_ResetCfg:
        cls = cast(Type[Reset], Module.parse(cfg["type"]))
        return cls.set_param(cfg, param_name, param_value)

    @staticmethod
    def auto_fill(cfg: Union[str, dict[str, Any]], ml: ModuleLibrary) -> ResetCfg:
        if isinstance(cfg, str):
            cfg = ml.get_module(cfg)

        cls = cast(Type[Reset], Module.parse(cfg["type"]))
        return cls.auto_fill(cfg, ml)


class NoneReset(AbsReset, tag="none"):
    def __init__(self, name: str, cfg: NoneResetCfg) -> None:
        self.name = name

    def init(self, prog: MyProgramV2) -> None: ...

    def total_length(self, _prog: MyProgramV2) -> Union[float, QickParam]:
        return 0.0

    def run(
        self, prog: MyProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        return t

    @staticmethod
    def set_param(
        _cfg: NoneResetCfg, _param_name: str, _param_value: Union[float, QickParam]
    ) -> NoneResetCfg:
        raise ValueError("NoneReset does not support set_param")

    @staticmethod
    def auto_fill(cfg: Union[str, dict[str, Any]], ml: ModuleLibrary) -> NoneResetCfg:
        if isinstance(cfg, str):
            cfg = ml.get_module(cfg)

        cfg["type"] = "reset/none"

        return cast(NoneResetCfg, cfg)


class PulseReset(AbsReset, tag="pulse"):
    def __init__(self, name: str, cfg: PulseResetCfg) -> None:
        self.name = name
        pulse_cfg = cfg["pulse_cfg"]
        self.reset_pulse = Pulse(name=f"{name}_pulse", cfg=pulse_cfg)

    def init(self, prog: MyProgramV2) -> None:
        self.reset_pulse.init(prog)

    def total_length(self, prog: MyProgramV2) -> Union[float, QickParam]:
        return self.reset_pulse.total_length(prog)

    def run(
        self, prog: MyProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        return self.reset_pulse.run(prog, t)

    @staticmethod
    def set_param(
        cfg: PulseResetCfg, param_name: str, param_value: Union[float, QickParam]
    ) -> PulseResetCfg:
        Pulse.set_param(cfg["pulse_cfg"], param_name, param_value)

        return cfg

    @staticmethod
    def auto_fill(cfg: Union[str, dict[str, Any]], ml: ModuleLibrary) -> PulseResetCfg:
        if isinstance(cfg, str):
            cfg = ml.get_module(cfg)

        cfg["type"] = "reset/pulse"
        cfg["pulse_cfg"] = Pulse.auto_fill(cfg["pulse_cfg"], ml)

        return cast(PulseResetCfg, cfg)


class TwoPulseReset(AbsReset, tag="two_pulse"):
    def __init__(self, name: str, cfg: TwoPulseResetCfg) -> None:
        self.name = name
        pulse1_cfg = cfg["pulse1_cfg"]
        pulse2_cfg = cfg["pulse2_cfg"]

        self.reset_pulse1 = Pulse(
            name=f"{name}_pulse1", cfg=pulse1_cfg, block_mode=False
        )
        self.reset_pulse2 = Pulse(name=f"{name}_pulse2", cfg=pulse2_cfg)

    def init(self, prog: MyProgramV2) -> None:
        self.reset_pulse1.init(prog)
        self.reset_pulse2.init(prog)

    def total_length(self, prog: MyProgramV2) -> Union[float, QickParam]:
        return calc_max_length(
            self.reset_pulse1.total_length(prog), self.reset_pulse2.total_length(prog)
        )

    def run(
        self, prog: MyProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        self.reset_pulse1.run(prog, t)
        self.reset_pulse2.run(prog, t)
        return t + self.total_length(prog)

    @staticmethod
    def set_param(
        cfg: TwoPulseResetCfg, param_name: str, param_value: Union[float, QickParam]
    ) -> TwoPulseResetCfg:
        if param_name == "on/off":
            Pulse.set_param(cfg["pulse1_cfg"], "on/off", param_value)
            Pulse.set_param(cfg["pulse2_cfg"], "on/off", param_value)
        elif param_name in ["gain1", "freq1"]:
            Pulse.set_param(cfg["pulse1_cfg"], param_name.replace("1", ""), param_value)
        elif param_name in ["gain2", "freq2"]:
            Pulse.set_param(cfg["pulse2_cfg"], param_name.replace("2", ""), param_value)
        elif param_name == "length":
            Pulse.set_param(cfg["pulse1_cfg"], "length", param_value)
            Pulse.set_param(cfg["pulse2_cfg"], "length", param_value)
        else:
            raise ValueError(f"Unknown parameter: {param_name}")

        return cfg

    @staticmethod
    def auto_fill(
        cfg: Union[str, dict[str, Any]], ml: ModuleLibrary
    ) -> TwoPulseResetCfg:
        if isinstance(cfg, str):
            cfg = ml.get_module(cfg)

        cfg["type"] = "reset/two_pulse"
        cfg["pulse1_cfg"] = Pulse.auto_fill(cfg["pulse1_cfg"], ml)
        cfg["pulse2_cfg"] = Pulse.auto_fill(cfg["pulse2_cfg"], ml)

        return cast(TwoPulseResetCfg, cfg)


class BathReset(AbsReset, tag="bath"):
    def __init__(self, name: str, cfg: BathResetCfg) -> None:
        self.name = name
        qubit_tone_cfg = cfg["qubit_tone_cfg"]
        cavity_tone_cfg = cfg["cavity_tone_cfg"]
        pi2_cfg = cfg["pi2_cfg"]

        self.qub_pulse = Pulse(
            name=f"{name}_qub_pulse", cfg=qubit_tone_cfg, block_mode=False
        )
        self.res_pulse = Pulse(name=f"{name}_res_pulse", cfg=cavity_tone_cfg)
        self.pi2_pulse = Pulse(name=f"{name}_pi2_pulse", cfg=pi2_cfg)

    def init(self, prog: MyProgramV2) -> None:
        self.qub_pulse.init(prog)
        self.res_pulse.init(prog)
        self.pi2_pulse.init(prog)

    def total_length(self, prog: MyProgramV2) -> Union[float, QickParam]:
        return calc_max_length(
            self.qub_pulse.total_length(prog),
            self.res_pulse.total_length(prog) + self.pi2_pulse.total_length(prog),
        )

    def run(
        self, prog: MyProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        self.qub_pulse.run(prog, t)  # non-blocking
        cur_t = self.res_pulse.run(prog, t)
        cur_t = self.pi2_pulse.run(prog, cur_t)

        return t + self.total_length(prog)

    @staticmethod
    def set_param(
        cfg: BathResetCfg, param_name: str, param_value: Union[float, QickParam]
    ) -> BathResetCfg:
        if param_name == "on/off":
            Pulse.set_param(cfg["qubit_tone_cfg"], "on/off", param_value)
            Pulse.set_param(cfg["cavity_tone_cfg"], "on/off", param_value)
            Pulse.set_param(cfg["pi2_cfg"], "on/off", param_value)
        elif param_name in ["qub_gain", "qub_freq"]:
            Pulse.set_param(
                cfg["qubit_tone_cfg"], param_name.replace("qub_", ""), param_value
            )
        elif param_name in ["res_gain", "res_freq"]:
            Pulse.set_param(
                cfg["cavity_tone_cfg"],
                param_name.replace("res_", ""),
                param_value,
            )
        elif param_name == "length":
            Pulse.set_param(cfg["qubit_tone_cfg"], "length", param_value)
            Pulse.set_param(cfg["cavity_tone_cfg"], "length", param_value)
        elif param_name == "pi2_phase":
            Pulse.set_param(cfg["pi2_cfg"], "phase", param_value)
        else:
            raise ValueError(f"Unknown parameter: {param_name}")

        return cfg

    @staticmethod
    def auto_fill(cfg: Union[str, dict[str, Any]], ml: ModuleLibrary) -> BathResetCfg:
        if isinstance(cfg, str):
            cfg = ml.get_module(cfg)

        cfg["type"] = "reset/bath"
        cfg["qubit_tone_cfg"] = Pulse.auto_fill(cfg["qubit_tone_cfg"], ml)
        cfg["cavity_tone_cfg"] = Pulse.auto_fill(cfg["cavity_tone_cfg"], ml)
        cfg["pi2_cfg"] = Pulse.auto_fill(cfg["pi2_cfg"], ml)

        return cast(BathResetCfg, cfg)
