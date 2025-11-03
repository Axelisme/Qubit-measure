from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Any, Callable, Dict, Type, TypeVar

from ..base import MyProgramV2
from .base import Module
from .pulse import Pulse, check_block_mode

T_Readout = TypeVar("T_Readout", bound="AbsReadout")


class TriggerReadout(Module):
    def __init__(
        self, name: str, ro_cfg: Dict[str, Any], gen_ch: int, gen_freq: float
    ) -> None:
        self.name = name
        self.ro_cfg = ro_cfg
        self.gen_ch = gen_ch
        self.gen_freq = gen_freq

    def init(self, prog: MyProgramV2) -> None:
        prog.declare_readout(ch=self.ro_cfg["ro_ch"], length=self.ro_cfg["ro_length"])
        prog.add_readoutconfig(
            ch=self.ro_cfg["ro_ch"],
            name=self.name,
            freq=self.ro_cfg.get("ro_freq", self.gen_freq),
            gen_ch=self.gen_ch,
        )

    def total_length(self) -> float:
        return self.ro_cfg["trig_offset"] + self.ro_cfg["ro_length"]

    def run(self, prog: MyProgramV2, t: float = 0.0) -> float:
        ro_ch: int = self.ro_cfg["ro_ch"]
        trig_offset: float = self.ro_cfg["trig_offset"]

        prog.send_readoutconfig(ro_ch, self.name, t=t)
        prog.trigger([ro_ch], t=t + trig_offset)

        return t + self.total_length()


class AbsReadout(Module):
    @classmethod
    def set_param(
        cls, readout_cfg: Dict[str, Any], param_name: str, param_value: float
    ) -> None:
        raise NotImplementedError(
            f"{cls.__name__} does not support set {param_name} params with {param_value}"
        )


support_readout_types: Dict[str, Type[AbsReadout]] = {}


def register_readout(ro_type: str) -> Callable[[T_Readout], T_Readout]:
    global support_readout_types

    if ro_type in support_readout_types:
        raise ValueError(
            f"Readout type {ro_type} already registered by {support_readout_types[ro_type].__name__}"
        )

    def decorator(cls: T_Readout) -> T_Readout:
        support_readout_types[ro_type] = cls
        return cls

    return decorator


def make_readout(name: str, readout_cfg: Dict[str, Any]) -> AbsReadout:
    ro_type = readout_cfg["type"]

    if ro_type not in support_readout_types:
        raise ValueError(f"Unknown readout type: {ro_type}")

    return support_readout_types[ro_type](name, cfg=readout_cfg)


def set_readout_cfg(
    readout_cfg: Dict[str, Any], param_name: str, param_value: float
) -> None:
    ro_type = readout_cfg["type"]

    if ro_type not in support_readout_types:
        raise ValueError(f"Unknown readout type: {ro_type}")

    support_readout_types[ro_type].set_param(readout_cfg, param_name, param_value)


@register_readout("base")
class BaseReadout(AbsReadout):
    def __init__(self, name: str, cfg: Dict[str, Any]) -> None:
        self.name = name
        self.pulse_cfg = deepcopy(cfg["pulse_cfg"])
        self.ro_cfg = deepcopy(cfg["ro_cfg"])

        ro_ch: int = self.pulse_cfg.setdefault("ro_ch", self.ro_cfg["ro_ch"])
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
        cls, readout_cfg: Dict[str, Any], param_name: str, param_value: float
    ) -> None:
        if param_name == "gain":
            readout_cfg["pulse_cfg"]["gain"] = param_value
        elif param_name == "freq":
            readout_cfg["pulse_cfg"]["freq"] = param_value
            # readout_cfg["ro_cfg"]["ro_freq"] = param_value
        elif param_name == "length":
            readout_cfg["pulse_cfg"]["waveform"]["length"] = param_value
        elif param_name == "ro_length":
            readout_cfg["ro_cfg"]["ro_length"] = param_value
        else:
            raise ValueError(f"Unknown parameter: {param_name}")

    def init(self, prog: MyProgramV2) -> None:
        self.pulse.init(prog)
        self.ro_trigger.init(prog)

    def run(self, prog: MyProgramV2, t: float = 0.0) -> float:
        self.ro_trigger.run(prog, t)
        self.pulse.run(prog, t)

        # use readout end as the new time
        ro_time = self.ro_trigger.total_length()
        pulse_time = self.pulse.total_length()

        if not (ro_time > pulse_time or ro_time < pulse_time):
            warnings.warn(
                f"Cannot determine the end time of {self.name}, this may cause unexpected behavior"
            )

        if ro_time > pulse_time:
            return t + ro_time
        else:
            return t + pulse_time
