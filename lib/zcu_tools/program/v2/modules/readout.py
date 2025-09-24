import warnings
from copy import deepcopy
from typing import Any, Dict

from ..base import MyProgramV2
from .base import Module
from .pulse import Pulse, check_block_mode


class AbsReadout(Module):
    @classmethod
    def set_param(
        cls, readout_cfg: Dict[str, Any], param_name: str, param_value: float
    ) -> None:
        raise NotImplementedError


def make_readout(name: str, readout_cfg: Dict[str, Any]) -> AbsReadout:
    ro_type = readout_cfg["type"]

    if ro_type == "base":
        return BaseReadout(
            name, pulse_cfg=readout_cfg["pulse_cfg"], ro_cfg=readout_cfg["ro_cfg"]
        )
    elif ro_type == "two_pulse":
        return TwoPulseReadout(
            name,
            pulse1_cfg=readout_cfg["pulse1_cfg"],
            pulse2_cfg=readout_cfg["pulse2_cfg"],
            ro_cfg=readout_cfg["ro_cfg"],
        )
    else:
        raise ValueError(f"Unknown readout type: {ro_type}")


def set_readout_cfg(
    readout_cfg: Dict[str, Any], param_name: str, param_value: float
) -> None:
    ro_type = readout_cfg["type"]

    if ro_type == "base":
        return BaseReadout.set_param(readout_cfg, param_name, param_value)
    elif ro_type == "two_pulse":
        return TwoPulseReadout.set_param(readout_cfg, param_name, param_value)
    else:
        raise ValueError(f"Unknown readout type: {ro_type}")


class BaseReadout(AbsReadout):
    def __init__(
        self, name: str, pulse_cfg: Dict[str, Any], ro_cfg: Dict[str, Any]
    ) -> None:
        self.name = name
        self.pulse_cfg = deepcopy(pulse_cfg)
        self.ro_cfg = deepcopy(ro_cfg)

        ro_ch: int = self.pulse_cfg.setdefault("ro_ch", ro_cfg["ro_ch"])
        if ro_ch != ro_cfg["ro_ch"]:
            warnings.warn(
                f"{name} pulse_cfg.ro_ch is {ro_ch}, this may not be what you want"
            )

        self.pulse = Pulse(name=f"{name}_pulse", cfg=self.pulse_cfg)

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

        prog.declare_readout(ch=self.ro_cfg["ro_ch"], length=self.ro_cfg["ro_length"])
        prog.add_readoutconfig(
            ch=self.ro_cfg["ro_ch"],
            name=f"{self.name}_adc",
            freq=self.ro_cfg.get("ro_freq", self.pulse_cfg["freq"]),
            gen_ch=self.pulse_cfg["ch"],
        )

    def run(self, prog: MyProgramV2, t: float = 0.0) -> float:
        ro_ch: int = self.ro_cfg["ro_ch"]

        prog.send_readoutconfig(ro_ch, f"{self.name}_adc", t=0)
        prog.trigger([ro_ch], t=t + self.ro_cfg["trig_offset"])

        t = self.pulse.run(prog, t)

        return t


class TwoPulseReadout(AbsReadout):
    def __init__(
        self,
        name: str,
        pulse1_cfg: Dict[str, Any],
        pulse2_cfg: Dict[str, Any],
        ro_cfg: Dict[str, Any],
    ) -> None:
        self.name = name
        self.ro_cfg = ro_cfg
        self.pulse1_cfg = deepcopy(pulse1_cfg)
        self.pulse2_cfg = deepcopy(pulse2_cfg)

        ro_ch: int = self.pulse1_cfg.setdefault("ro_ch", ro_cfg["ro_ch"])
        if ro_ch != ro_cfg["ro_ch"]:
            warnings.warn(
                f"{name} pulse1_cfg.ro_ch is {ro_ch}, this may not be what you want"
            )

        ro_ch: int = self.pulse2_cfg.setdefault("ro_ch", ro_cfg["ro_ch"])
        if ro_ch != ro_cfg["ro_ch"]:
            warnings.warn(
                f"{name} pulse2_cfg.ro_ch is {ro_ch}, this may not be what you want"
            )

        self.pulse1 = Pulse(name=f"{name}_pulse1", cfg=self.pulse1_cfg)
        self.pulse2 = Pulse(name=f"{name}_pulse2", cfg=self.pulse2_cfg)

        check_block_mode(self.pulse1.name, self.pulse1_cfg, want_block=True)
        check_block_mode(self.pulse2.name, self.pulse2_cfg, want_block=True)

    @classmethod
    def set_param(
        cls, readout_cfg: Dict[str, Any], param_name: str, param_value: float
    ) -> None:
        if param_name == "gain":
            readout_cfg["pulse2_cfg"]["gain"] = param_value
        elif param_name == "freq":
            readout_cfg["pulse1_cfg"]["freq"] = param_value
            readout_cfg["pulse2_cfg"]["freq"] = param_value
        elif param_name == "length":
            readout_cfg["pulse2_cfg"]["waveform"]["length"] = param_value
        elif param_name == "ro_length":
            readout_cfg["ro_cfg"]["ro_length"] = param_value
        else:
            raise ValueError(f"Unknown parameter: {param_name}")

    def init(self, prog: MyProgramV2) -> None:
        self.pulse1.init(prog)
        self.pulse2.init(prog)

        prog.declare_readout(ch=self.ro_cfg["ro_ch"], length=self.ro_cfg["ro_length"])
        prog.add_readoutconfig(
            ch=self.ro_cfg["ro_ch"],
            name=f"{self.name}_adc",
            freq=self.ro_cfg.get("ro_freq", self.pulse2_cfg["freq"]),
            gen_ch=self.pulse2_cfg["ch"],
        )

    def run(self, prog: MyProgramV2, t: float = 0.0) -> float:
        ro_ch: int = self.ro_cfg["ro_ch"]

        prog.send_readoutconfig(ro_ch, f"{self.name}_adc", t=0)
        prog.trigger([ro_ch], t=t + self.ro_cfg["trig_offset"])

        t = self.pulse1.run(prog, t)
        t = self.pulse2.run(prog, t)

        return t
