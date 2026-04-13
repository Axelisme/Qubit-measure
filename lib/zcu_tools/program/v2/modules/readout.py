from __future__ import annotations

import warnings
from abc import abstractmethod
from copy import deepcopy

from qick.asm_v2 import QickParam
from typing_extensions import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Literal,
    Optional,
    Self,
    TypeAlias,
    Union,
)

from ..base import MyProgramV2
from .base import Module, ModuleCfg
from .pulse import Pulse, PulseCfg
from .util import calc_max_length, round_timestamp

if TYPE_CHECKING:
    from zcu_tools.meta_tool import ModuleLibrary


@ModuleCfg.register_handler("readout/direct")
class DirectReadoutCfg(ModuleCfg):
    type: Literal["readout/direct"] = "readout/direct"
    ro_ch: int
    ro_length: Union[float, QickParam]
    ro_freq: Union[float, QickParam]
    trig_offset: Union[float, QickParam] = 0.0
    gen_ch: Optional[int] = None

    def set_param(self, name: str, value: Union[float, QickParam]) -> None:
        if name == "ro_freq":
            self.ro_freq = value
        elif name == "ro_length":
            self.ro_length = value
        else:
            raise ValueError(f"Unknown parameter: {name}")


@ModuleCfg.register_handler("readout/pulse")
class PulseReadoutCfg(ModuleCfg):
    type: Literal["readout/pulse"] = "readout/pulse"
    pulse_cfg: PulseCfg
    ro_cfg: DirectReadoutCfg

    @classmethod
    def from_dict(cls, raw_cfg: dict[str, Any], ml: "ModuleLibrary") -> Self:
        raw_cfg = deepcopy(raw_cfg)

        pulse_cfg = raw_cfg.get("pulse_cfg")
        if isinstance(pulse_cfg, str):
            raw_cfg["pulse_cfg"] = ml.get_module(pulse_cfg)
        ro_cfg = raw_cfg.get("ro_cfg")
        if isinstance(ro_cfg, str):
            raw_cfg["ro_cfg"] = ml.get_module(ro_cfg)

        # auto derive ro_ch/ro_freq from pulse_cfg.ch/freq
        if isinstance(pulse_cfg := raw_cfg.get("pulse_cfg"), dict):
            ch = pulse_cfg.get("ch")
            freq = pulse_cfg.get("freq")
            if isinstance(ro_cfg := raw_cfg.get("ro_cfg"), dict):
                ro_cfg.setdefault("ro_ch", ch)
                ro_cfg.setdefault("ro_freq", freq)

        return cls.model_validate(raw_cfg)

    def set_param(self, name: str, value: Union[float, QickParam]) -> None:
        if name == "gain":
            self.pulse_cfg.set_param("gain", value)
        elif name == "freq":
            self.pulse_cfg.set_param("freq", value)
            self.ro_cfg.set_param("ro_freq", value)
        elif name == "ro_freq":
            self.ro_cfg.set_param("ro_freq", value)
        elif name == "length":
            self.pulse_cfg.set_param("length", value)
        elif name == "ro_length":
            self.ro_cfg.set_param("ro_length", value)
        else:
            raise ValueError(f"Unknown parameter: {name}")


ReadoutCfg: TypeAlias = Union[PulseReadoutCfg, DirectReadoutCfg]


class AbsReadout(Module):
    @abstractmethod
    def total_length(self, prog: MyProgramV2) -> Union[float, QickParam]: ...


class Readout(AbsReadout):
    _supported_readout: ClassVar[dict[str, type["AbsReadout"]]] = {}

    def __init__(self, name: str, cfg: ReadoutCfg) -> None:
        cfg_type = cfg.type
        if cfg_type not in self._supported_readout:
            raise ValueError(f"Unknown readout type: {cfg_type}")
        self.readout = self._supported_readout[cfg_type](name, cfg)

    @classmethod
    def register_readout(
        cls, id_name: str
    ) -> Callable[[type["AbsReadout"]], type["AbsReadout"]]:
        if id_name in cls._supported_readout:
            raise ValueError(
                f"Readout {id_name} already registered by {cls._supported_readout[id_name].__name__}"
            )

        def decorator(sub_cls: type["AbsReadout"]) -> type["AbsReadout"]:
            cls._supported_readout[id_name] = sub_cls
            return sub_cls

        return decorator

    @property
    def name(self) -> str:
        return self.readout.name

    def init(self, prog: MyProgramV2) -> None:
        self.readout.init(prog)

    def total_length(self, prog: MyProgramV2) -> Union[float, QickParam]:
        return self.readout.total_length(prog)

    def run(
        self, prog: MyProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        return self.readout.run(prog, t)

@Readout.register_readout("readout/direct")
class DirectReadout(AbsReadout):
    def __init__(self, name: str, cfg: DirectReadoutCfg) -> None:
        self.name = name
        self.cfg = deepcopy(cfg)

    def init(self, prog: MyProgramV2) -> None:
        prog.declare_readout(ch=self.cfg.ro_ch, length=self.cfg.ro_length)
        prog.add_readoutconfig(
            ch=self.cfg.ro_ch,
            name=self.name,
            freq=self.cfg.ro_freq,
            gen_ch=self.cfg.gen_ch,
        )

    def total_length(self, prog: MyProgramV2) -> Union[float, QickParam]:
        return round_timestamp(
            prog,
            round_timestamp(prog, self.cfg.trig_offset)
            + round_timestamp(prog, self.cfg.ro_length, ro_ch=self.cfg.ro_ch),
        )

    def run(
        self, prog: MyProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        ro_ch = self.cfg.ro_ch
        trig_offset = self.cfg.trig_offset
        prog.send_readoutconfig(ro_ch, self.name, t=t)  # type: ignore
        prog.trigger([ro_ch], t=t + trig_offset)
        return t

@Readout.register_readout("readout/pulse")
class PulseReadout(AbsReadout):
    def __init__(self, name: str, cfg: PulseReadoutCfg) -> None:
        self.name = name
        self.cfg = deepcopy(cfg)
        ro_ch = self.cfg.pulse_cfg.ro_ch
        if ro_ch is None:
            ro_ch = self.cfg.ro_cfg.ro_ch
            self.cfg.pulse_cfg.ro_ch = ro_ch
        if ro_ch != self.cfg.ro_cfg.ro_ch:
            warnings.warn(
                f"{name} pulse_cfg.ro_ch is {ro_ch}, this may not be what you want"
            )
        self.pulse = Pulse(name=f"{name}_pulse", cfg=self.cfg.pulse_cfg)
        self.ro_window = DirectReadout(name=f"{name}_adc", cfg=self.cfg.ro_cfg)

    def init(self, prog: MyProgramV2) -> None:
        self.pulse.init(prog)
        self.ro_window.init(prog)

    def total_length(self, prog: MyProgramV2) -> Union[float, QickParam]:
        return calc_max_length(
            self.ro_window.total_length(prog), self.pulse.total_length(prog)
        )

    def run(
        self, prog: MyProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        self.ro_window.run(prog, t)
        self.pulse.run(prog, t)
        return t + self.total_length(prog)
