from typing import Any, Dict
from copy import deepcopy

from ..base import force_no_post_delay
from .base import Module
from .pulse import MyProgramV2, Pulse


class AbsReadout(Module):
    pass


def make_readout(name: str, readout_cfg: Dict[str, Any]) -> AbsReadout:
    ro_type = readout_cfg["type"]

    if ro_type == "base":
        return BaseReadout(
            name,
            pulse_cfg=readout_cfg["pulse_cfg"],
            ro_cfg=readout_cfg["ro_cfg"],
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


class BaseReadout(AbsReadout):
    def __init__(
        self,
        name: str,
        pulse_cfg: Dict[str, Any],
        ro_cfg: Dict[str, Any],
    ) -> None:
        self.name = name
        self.pulse_cfg = deepcopy(pulse_cfg)
        self.ro_cfg = deepcopy(ro_cfg)

        # TODO: support post delay
        pulse_name = f"{name}_pulse"
        force_no_post_delay(self.pulse_cfg, pulse_name)
        self.pulse = Pulse(
            name=pulse_name, cfg=self.pulse_cfg, ro_ch=self.ro_cfg["ro_ch"]
        )

    def init(self, prog: MyProgramV2) -> None:
        self.pulse.init(prog)

        prog.declare_readout(ch=self.ro_cfg["ro_ch"], length=self.ro_cfg["ro_length"])
        prog.add_readoutconfig(
            ch=self.ro_cfg["ro_ch"],
            name=f"{self.name}_readout_adc",
            freq=self.ro_cfg.get("ro_freq", self.pulse_cfg["freq"]),
            gen_ch=self.pulse_cfg["ch"],
        )

    def run(self, prog: MyProgramV2) -> None:
        ro_ch: int = self.ro_cfg["ro_ch"]

        prog.send_readoutconfig(ro_ch, f"{self.name}_readout_adc", t=0)

        self.pulse.run(prog)

        prog.trigger([ro_ch], t=self.ro_cfg["trig_offset"])


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

        # TODO: support post delay
        pulse2_name = f"{name}_pulse2"
        force_no_post_delay(self.pulse2_cfg, pulse2_name)

        self.pulse1 = Pulse(name=f"{name}_pulse1", cfg=self.pulse1_cfg)
        self.pulse2 = Pulse(
            name=f"{name}_pulse2", cfg=self.pulse2_cfg, ro_ch=self.ro_cfg["ro_ch"]
        )

    def init(self, prog: MyProgramV2) -> None:
        self.pulse1.init(prog)
        self.pulse2.init(prog)

        prog.declare_readout(ch=self.ro_cfg["ro_ch"], length=self.ro_cfg["ro_length"])
        prog.add_readoutconfig(
            ch=self.ro_cfg["ro_ch"],
            name=f"{self.name}_readout_adc",
            freq=self.ro_cfg.get("ro_freq", self.pulse2_cfg["freq"]),
            gen_ch=self.pulse2_cfg["ch"],
        )

    def run(self, prog: MyProgramV2) -> None:
        ro_ch: int = self.ro_cfg["ro_ch"]

        prog.send_readoutconfig(ro_ch, f"{self.name}_readout_adc", t=0)

        self.pulse1.run(prog)
        self.pulse2.run(prog)

        prog.trigger([ro_ch], t=self.ro_cfg["trig_offset"])
