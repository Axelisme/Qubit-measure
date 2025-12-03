from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Callable, ClassVar, Dict, Optional, Type, TypedDict, Union

from qick.asm_v2 import QickParam

from ..base import MyProgramV2
from .base import Module
from .pulse import Pulse, PulseCfg, check_block_mode
from .util import calc_max_length


class TriggerCfg(TypedDict):
    ro_ch: int
    ro_length: Union[float, QickParam]
    ro_freq: Union[float, QickParam]
    trig_offset: Union[float, QickParam]


class TriggerReadout(Module):
    def __init__(
        self,
        name: str,
        ro_cfg: TriggerCfg,
        gen_ch: int,
        gen_freq: Optional[Union[float, QickParam]] = None,
    ) -> None:
        self.name = name
        self.ro_cfg = ro_cfg
        self.gen_ch = gen_ch
        self.gen_freq = gen_freq

    def init(self, prog: MyProgramV2) -> None:
        prog.declare_readout(ch=self.ro_cfg["ro_ch"], length=self.ro_cfg["ro_length"])

        ro_freq = self.ro_cfg.get("ro_freq", self.gen_freq)
        if ro_freq is None:
            raise ValueError("ro_freq is not set in ro_cfg or gen_freq is not set")

        prog.add_readoutconfig(
            ch=self.ro_cfg["ro_ch"],
            name=self.name,
            freq=ro_freq,
            gen_ch=self.gen_ch,
        )

    def total_length(self) -> Union[float, QickParam]:
        return self.ro_cfg["trig_offset"] + self.ro_cfg["ro_length"]

    def run(
        self, prog: MyProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        ro_ch = self.ro_cfg["ro_ch"]
        trig_offset = self.ro_cfg["trig_offset"]

        prog.send_readoutconfig(ro_ch, self.name, t=t)
        prog.trigger([ro_ch], t=t + trig_offset)

        return t  # always non-blocking


class BaseReadoutCfg(TypedDict):
    type: str
    pulse_cfg: PulseCfg
    ro_cfg: TriggerCfg


ReadoutCfg = Union[BaseReadoutCfg]


class AbsReadout(Module):
    def __init__(self, name: str, cfg: ReadoutCfg) -> None: ...

    @classmethod
    def set_param(
        cls,
        readout_cfg: ReadoutCfg,
        param_name: str,
        param_value: Union[float, QickParam],
    ) -> ReadoutCfg:
        raise NotImplementedError(
            f"{cls.__name__} does not support set {param_name} params with {param_value}"
        )

    def total_length(self) -> Union[float, QickParam]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support total_length"
        )


class Readout(AbsReadout):
    SUPPORTED_TYPES: ClassVar[Dict[str, Type[AbsReadout]]] = {}

    @classmethod
    def register_readout(
        cls, ro_type: str
    ) -> Callable[[Type[AbsReadout]], Type[AbsReadout]]:
        if ro_type in cls.SUPPORTED_TYPES:
            raise ValueError(
                f"Readout type {ro_type} already registered by {cls.SUPPORTED_TYPES[ro_type].__name__}"
            )

        def decorator(cls: Type[AbsReadout]) -> Type[AbsReadout]:
            Readout.SUPPORTED_TYPES[ro_type] = cls
            return cls

        return decorator

    @classmethod
    def get_readout_cls(cls, readout_cfg: ReadoutCfg) -> Type[AbsReadout]:
        if readout_cfg["type"] not in cls.SUPPORTED_TYPES:
            raise ValueError(f"Unknown readout type: {readout_cfg['type']}")
        return cls.SUPPORTED_TYPES[readout_cfg["type"]]

    @classmethod
    def set_param(
        cls,
        readout_cfg: ReadoutCfg,
        param_name: str,
        param_value: Union[float, QickParam],
    ) -> ReadoutCfg:
        return cls.get_readout_cls(readout_cfg).set_param(
            readout_cfg, param_name, param_value
        )

    def __init__(self, name: str, cfg: ReadoutCfg) -> None:
        self.name = name
        self.cfg = deepcopy(cfg)
        readout_cls = self.get_readout_cls(cfg)
        self.readout = readout_cls(name, cfg)

    def init(self, prog: MyProgramV2) -> None:
        self.readout.init(prog)

    def run(
        self, prog: MyProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        return self.readout.run(prog, t)

    def total_length(self) -> Union[float, QickParam]:
        return self.readout.total_length()


@Readout.register_readout("base")
class BaseReadout(AbsReadout):
    def __init__(self, name: str, cfg: BaseReadoutCfg) -> None:
        self.name = name
        self.pulse_cfg = deepcopy(cfg["pulse_cfg"])
        self.ro_cfg = deepcopy(cfg["ro_cfg"])

        ro_ch = self.pulse_cfg.setdefault("ro_ch", self.ro_cfg["ro_ch"])
        if ro_ch != self.ro_cfg["ro_ch"]:
            warnings.warn(
                f"{name} pulse_cfg.ro_ch is {ro_ch}, this may not be what you want"
            )

        self.pulse = Pulse(name=f"{name}_pulse", cfg=self.pulse_cfg)
        self.ro_trigger = TriggerReadout(
            name=f"{name}_adc",
            ro_cfg=self.ro_cfg,
            gen_ch=self.pulse_cfg["ch"],
            gen_freq=self.pulse_cfg["freq"],
        )

        check_block_mode(self.pulse.name, self.pulse_cfg, want_block=True)

    @classmethod
    def set_param(
        cls,
        readout_cfg: BaseReadoutCfg,
        param_name: str,
        param_value: Union[float, QickParam],
    ) -> BaseReadoutCfg:
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
        self.ro_trigger.init(prog)

    def total_length(self) -> Union[float, QickParam]:
        return calc_max_length(
            self.ro_trigger.total_length(), self.pulse.total_length()
        )

    def run(
        self, prog: MyProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        t = self.ro_trigger.run(prog, t)
        t = self.pulse.run(prog, t)

        return t + self.total_length()
