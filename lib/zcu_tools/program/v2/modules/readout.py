from __future__ import annotations

import warnings
from copy import deepcopy

from qick.asm_v2 import QickParam
from typing_extensions import TYPE_CHECKING, Any, Dict, NotRequired, Type, Union, cast

from ..base import MyProgramV2
from .base import Module, ModuleCfg
from .pulse import Pulse, PulseCfg, check_block_mode
from .util import calc_max_length

if TYPE_CHECKING:
    from zcu_tools.meta_manager import ModuleLibrary


class Readout(Module, tag="readout"):
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
        readout_cfg: ReadoutCfg,
        param_name: str,
        param_value: Union[float, QickParam],
    ) -> ReadoutCfg:
        return cast(Type[Readout], Module.parse(readout_cfg["type"])).set_param(
            readout_cfg, param_name, param_value
        )

    @staticmethod
    def auto_fill(
        readout_cfg: Union[str, Dict[str, Any]], ml: ModuleLibrary
    ) -> ReadoutCfg:
        if isinstance(readout_cfg, str):
            readout_cfg = ml.get_module(readout_cfg)

        return cast(Type[Readout], Module.parse(readout_cfg["type"])).auto_fill(
            readout_cfg, ml
        )


class DirectReadoutCfg(ModuleCfg, closed=True):
    ro_ch: int
    ro_length: Union[float, QickParam]
    ro_freq: Union[float, QickParam]
    trig_offset: Union[float, QickParam]

    gen_ch: NotRequired[int]


class DirectReadout(Readout, tag="direct"):
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
    def auto_fill(
        cls, readout_cfg: Union[str, Dict[str, Any]], ml: ModuleLibrary
    ) -> DirectReadoutCfg:
        if isinstance(readout_cfg, str):
            readout_cfg = ml.get_module(readout_cfg)

        readout_cfg.setdefault("trig_offset", 0.0)

        return cast(DirectReadoutCfg, readout_cfg)


class PulseReadoutCfg(ModuleCfg, closed=True):
    pulse_cfg: PulseCfg
    ro_cfg: DirectReadoutCfg


class PulseReadout(Readout, tag="pulse"):
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
        cls,
        readout_cfg: PulseReadoutCfg,
        param_name: str,
        param_value: Union[float, QickParam],
    ) -> PulseReadoutCfg:
        if param_name == "gain":
            readout_cfg["pulse_cfg"]["gain"] = param_value
        elif param_name == "freq":
            readout_cfg["pulse_cfg"]["freq"] = param_value
            readout_cfg["ro_cfg"]["ro_freq"] = param_value
        elif param_name == "length":
            readout_cfg["pulse_cfg"]["waveform"]["length"] = param_value
        elif param_name == "ro_length":
            readout_cfg["ro_cfg"]["ro_length"] = param_value
        else:
            raise ValueError(f"Unknown parameter: {param_name}")

        return readout_cfg

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

    @classmethod
    def auto_fill(
        cls, readout_cfg: Union[str, Dict[str, Any]], ml: ModuleLibrary
    ) -> PulseReadoutCfg:
        if isinstance(readout_cfg, str):
            readout_cfg = ml.get_module(readout_cfg)

        readout_cfg["pulse_cfg"] = Pulse.auto_fill(readout_cfg["pulse_cfg"], ml)
        readout_cfg["ro_cfg"] = DirectReadout.auto_fill(readout_cfg["ro_cfg"], ml)
        if freq := readout_cfg["pulse_cfg"].get("freq"):
            readout_cfg["ro_cfg"].setdefault("ro_freq", freq)
        readout_cfg["ro_cfg"].setdefault("gen_ch", readout_cfg["pulse_cfg"]["ch"])

        return cast(PulseReadoutCfg, readout_cfg)


ReadoutCfg = Union[PulseReadoutCfg, DirectReadoutCfg]
