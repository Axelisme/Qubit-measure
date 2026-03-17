from __future__ import annotations

import warnings
from copy import deepcopy

from qick.asm_v2 import QickParam
from typing_extensions import (
    TYPE_CHECKING,
    Any,
    NotRequired,
    Type,
    TypeAlias,
    TypeVar,
    Union,
    cast,
)

from ..base import MyProgramV2
from .base import Module, ModuleCfg
from .pulse import Pulse, PulseCfg, Waveform, check_block_mode
from .util import calc_max_length

if TYPE_CHECKING:
    from zcu_tools.meta_tool import ModuleLibrary


class DirectReadoutCfg(ModuleCfg, closed=True):
    ro_ch: int
    ro_length: Union[float, QickParam]
    ro_freq: Union[float, QickParam]
    trig_offset: Union[float, QickParam]

    gen_ch: NotRequired[int]


class PulseReadoutCfg(ModuleCfg, closed=True):
    pulse_cfg: PulseCfg
    ro_cfg: DirectReadoutCfg


ReadoutCfg: TypeAlias = Union[PulseReadoutCfg, DirectReadoutCfg]

T_ReadoutCfg = TypeVar("T_ReadoutCfg", bound=ReadoutCfg)


class AbsReadout(Module, tag="readout"): ...


class Readout(Module):
    def __init__(self, name: str, cfg: ReadoutCfg) -> None:
        self.name = name
        self.readout = cast(Readout, Module.parse(cfg["type"])(name, cfg))

    def init(self, prog: MyProgramV2) -> None:
        self.readout.init(prog)

    def total_length(self) -> Union[float, QickParam]:
        return self.readout.total_length()

    def run(
        self, prog: MyProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        return self.readout.run(prog, t)

    @staticmethod
    def set_param(
        cfg: T_ReadoutCfg, param_name: str, param_value: Union[float, QickParam]
    ) -> T_ReadoutCfg:
        return cast(Type[Readout], Module.parse(cfg["type"])).set_param(
            cfg, param_name, param_value
        )

    @staticmethod
    def auto_fill(cfg: Union[str, dict[str, Any]], ml: ModuleLibrary) -> ReadoutCfg:
        if isinstance(cfg, str):
            cfg = ml.get_module(cfg)

        return cast(Type[Readout], Module.parse(cfg["type"])).auto_fill(cfg, ml)


class DirectReadout(AbsReadout, tag="direct"):
    def __init__(self, name: str, cfg: DirectReadoutCfg) -> None:
        self.name = name
        self.cfg = cfg

    def init(self, prog: MyProgramV2) -> None:
        prog.declare_readout(ch=self.cfg["ro_ch"], length=self.cfg["ro_length"])

        prog.add_readoutconfig(
            ch=self.cfg["ro_ch"],
            name=self.name,
            freq=self.cfg["ro_freq"],
            gen_ch=self.cfg.get("gen_ch"),
        )

    def total_length(self) -> Union[float, QickParam]:
        return self.cfg["trig_offset"] + self.cfg["ro_length"]

    def run(
        self, prog: MyProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        ro_ch = self.cfg["ro_ch"]
        trig_offset = self.cfg["trig_offset"]

        prog.send_readoutconfig(ro_ch, self.name, t=t)  # type: ignore
        prog.trigger([ro_ch], t=t + trig_offset)

        return t  # always non-blocking

    @classmethod
    def set_param(
        cls,
        cfg: DirectReadoutCfg,
        param_name: str,
        param_value: Union[float, QickParam],
    ) -> DirectReadoutCfg:
        if param_name == "ro_freq":
            cfg["ro_freq"] = param_value
        elif param_name == "ro_length":
            cfg["ro_length"] = param_value
        else:
            raise ValueError(f"Unknown parameter: {param_name}")

        return cfg

    @staticmethod
    def auto_fill(
        cfg: Union[str, dict[str, Any]], ml: ModuleLibrary
    ) -> DirectReadoutCfg:
        if isinstance(cfg, str):
            cfg = ml.get_module(cfg)

        cfg["type"] = "readout/direct"
        cfg.setdefault("trig_offset", 0.0)

        return cast(DirectReadoutCfg, cfg)


class PulseReadout(AbsReadout, tag="pulse"):
    def __init__(self, name: str, cfg: PulseReadoutCfg) -> None:
        self.name = name
        self.pulse_cfg = deepcopy(cfg["pulse_cfg"])
        self.ro_cfg = deepcopy(cfg["ro_cfg"])

        ro_ch = self.pulse_cfg.setdefault("ro_ch", self.ro_cfg["ro_ch"])
        if ro_ch != self.ro_cfg["ro_ch"]:
            warnings.warn(
                f"{name} pulse_cfg.ro_ch is {ro_ch}, this may not be what you want"
            )

        self.pulse = Pulse(name=f"{name}_pulse", cfg=self.pulse_cfg)
        self.ro_window = DirectReadout(name=f"{name}_adc", cfg=self.ro_cfg)

        check_block_mode(self.pulse.name, self.pulse_cfg, want_block=True)

    @classmethod
    def set_param(
        cls, cfg: PulseReadoutCfg, param_name: str, param_value: Union[float, QickParam]
    ) -> PulseReadoutCfg:
        if param_name == "gain":
            Pulse.set_param(cfg["pulse_cfg"], "gain", param_value)
        elif param_name == "freq":
            Pulse.set_param(cfg["pulse_cfg"], "freq", param_value)
            DirectReadout.set_param(cfg["ro_cfg"], "ro_freq", param_value)
        elif param_name == "ro_freq":
            DirectReadout.set_param(cfg["ro_cfg"], "ro_freq", param_value)
        elif param_name == "length":
            Waveform.set_param(cfg["pulse_cfg"]["waveform"], "length", param_value)
        elif param_name == "ro_length":
            DirectReadout.set_param(cfg["ro_cfg"], "ro_length", param_value)
        else:
            raise ValueError(f"Unknown parameter: {param_name}")

        return cfg

    def init(self, prog: MyProgramV2) -> None:
        self.pulse.init(prog)
        self.ro_window.init(prog)

    def total_length(self) -> Union[float, QickParam]:
        return calc_max_length(self.ro_window.total_length(), self.pulse.total_length())

    def run(
        self, prog: MyProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        t = self.ro_window.run(prog, t)
        t = self.pulse.run(prog, t)

        return t + self.total_length()

    @staticmethod
    def auto_fill(
        cfg: Union[str, dict[str, Any]], ml: ModuleLibrary
    ) -> PulseReadoutCfg:
        if isinstance(cfg, str):
            cfg = ml.get_module(cfg)

        cfg["type"] = "readout/pulse"
        cfg["pulse_cfg"] = Pulse.auto_fill(cfg["pulse_cfg"], ml)
        cfg["ro_cfg"] = DirectReadout.auto_fill(cfg["ro_cfg"], ml)
        if (freq := cfg["pulse_cfg"].get("freq")) is not None:
            cfg["ro_cfg"].setdefault("ro_freq", freq)
        cfg["ro_cfg"].setdefault("gen_ch", cfg["pulse_cfg"]["ch"])

        return cast(PulseReadoutCfg, cfg)
