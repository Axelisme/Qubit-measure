from __future__ import annotations

from qick.asm_v2 import QickParam
from typing_extensions import TYPE_CHECKING, Any, Dict, Optional, Type, Union, cast

from ..base import MyProgramV2
from .base import Module, ModuleCfg
from .pulse import Pulse, PulseCfg, check_block_mode
from .util import calc_max_length

if TYPE_CHECKING:
    from zcu_tools.meta_manager import ModuleLibrary


class Reset(Module, tag="reset"):
    def __init__(self, name: str, cfg: Optional[ResetCfg]) -> None:
        self.name = name

        if cfg is None:
            cfg = {"type": "reset/none"}

        self.reset = cast(Reset, Module.parse(cfg["type"])(name, cfg))

    def init(self, prog: MyProgramV2) -> None:
        self.reset.init(prog)

    def total_length(self) -> Union[float, QickParam]:
        return self.reset.total_length()

    def run(
        self, prog: MyProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        return self.reset.run(prog, t)

    @staticmethod
    def set_param(
        reset_cfg: ResetCfg,
        param_name: str,
        param_value: Union[float, QickParam],
    ) -> ResetCfg:
        cls = cast(Type[Reset], Module.parse(reset_cfg["type"]))
        return cls.set_param(reset_cfg, param_name, param_value)

    @staticmethod
    def auto_fill(reset_cfg: Union[str, Dict[str, Any]], ml: ModuleLibrary) -> ResetCfg:
        if isinstance(reset_cfg, str):
            reset_cfg = ml.get_module(reset_cfg)

        cls = cast(Type[Reset], Module.parse(reset_cfg["type"]))
        return cls.auto_fill(reset_cfg, ml)


class NoneResetCfg(ModuleCfg): ...


class NoneReset(Reset, tag="none"):
    def __init__(self, name: str, cfg: NoneResetCfg) -> None:
        self.name = name

    def init(self, prog: MyProgramV2) -> None: ...

    def total_length(self) -> Union[float, QickParam]:
        return 0.0

    def run(
        self, prog: MyProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        return t

    @staticmethod
    def set_param(
        reset_cfg: NoneResetCfg, param_name: str, param_value: Union[float, QickParam]
    ) -> NoneResetCfg:
        raise ValueError("NoneReset does not support set_param")

    @staticmethod
    def auto_fill(
        reset_cfg: Union[str, Dict[str, Any]], ml: ModuleLibrary
    ) -> NoneResetCfg:
        if isinstance(reset_cfg, str):
            reset_cfg = ml.get_module(reset_cfg)

        return cast(NoneResetCfg, reset_cfg)


class PulseResetCfg(ModuleCfg):
    pulse_cfg: PulseCfg


class PulseReset(Reset, tag="pulse"):
    def __init__(self, name: str, cfg: PulseResetCfg) -> None:
        self.name = name
        pulse_cfg = cfg["pulse_cfg"]
        self.reset_pulse = Pulse(name=f"{name}_pulse", cfg=pulse_cfg)
        check_block_mode(self.reset_pulse.name, pulse_cfg, want_block=True)

    def init(self, prog: MyProgramV2) -> None:
        self.reset_pulse.init(prog)

    def total_length(self) -> Union[float, QickParam]:
        return self.reset_pulse.total_length()

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

    @staticmethod
    def auto_fill(
        reset_cfg: Union[str, Dict[str, Any]], ml: ModuleLibrary
    ) -> PulseResetCfg:
        if isinstance(reset_cfg, str):
            reset_cfg = ml.get_module(reset_cfg)

        reset_cfg["pulse_cfg"] = Pulse.auto_fill(reset_cfg["pulse_cfg"], ml)

        return cast(PulseResetCfg, reset_cfg)


class TwoPulseResetCfg(ModuleCfg):
    pulse1_cfg: PulseCfg
    pulse2_cfg: PulseCfg


class TwoPulseReset(Reset, tag="two_pulse"):
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

    def total_length(self) -> Union[float, QickParam]:
        return calc_max_length(
            self.reset_pulse1.total_length(), self.reset_pulse2.total_length()
        )

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

    @staticmethod
    def auto_fill(
        reset_cfg: Union[str, Dict[str, Any]], ml: ModuleLibrary
    ) -> TwoPulseResetCfg:
        if isinstance(reset_cfg, str):
            reset_cfg = ml.get_module(reset_cfg)

        reset_cfg["pulse1_cfg"] = Pulse.auto_fill(reset_cfg["pulse1_cfg"], ml)
        reset_cfg["pulse2_cfg"] = Pulse.auto_fill(reset_cfg["pulse2_cfg"], ml)

        return cast(TwoPulseResetCfg, reset_cfg)


class BathResetCfg(ModuleCfg):
    qubit_tone_cfg: PulseCfg
    cavity_tone_cfg: PulseCfg
    pi2_cfg: PulseCfg


class BathReset(Reset, tag="bath"):
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

    def total_length(self) -> Union[float, QickParam]:
        max_length = calc_max_length(
            self.qub_pulse.total_length(), self.res_pulse.total_length()
        )
        return max_length + self.pi2_pulse.total_length()

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

    @staticmethod
    def auto_fill(
        reset_cfg: Union[str, Dict[str, Any]], ml: ModuleLibrary
    ) -> BathResetCfg:
        if isinstance(reset_cfg, str):
            reset_cfg = ml.get_module(reset_cfg)

        reset_cfg["qubit_tone_cfg"] = Pulse.auto_fill(reset_cfg["qubit_tone_cfg"], ml)
        reset_cfg["cavity_tone_cfg"] = Pulse.auto_fill(reset_cfg["cavity_tone_cfg"], ml)
        reset_cfg["pi2_cfg"] = Pulse.auto_fill(reset_cfg["pi2_cfg"], ml)

        return cast(BathResetCfg, reset_cfg)


ResetCfg = Union[NoneResetCfg, PulseResetCfg, TwoPulseResetCfg, BathResetCfg]
