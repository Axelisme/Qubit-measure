from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Optional

from .pulse import declare_pulse, set_pulse

if TYPE_CHECKING:
    from .program import MyProgramV1


def make_readout(name: str) -> AbsReadout:
    if name == "base":
        return BaseReadout()
    else:
        raise ValueError(f"Unknown readout type: {name}")


class AbsReadout(ABC):
    @abstractmethod
    def init(self, prog: MyProgramV1) -> None:
        pass

    @abstractmethod
    def readout_qubit(self, prog: MyProgramV1) -> None:
        pass


class BaseReadout(AbsReadout):
    def init(self, prog: MyProgramV1) -> None:
        res_ch = prog.res_pulse["ch"]
        ro_chs = prog.adc["chs"]

        declare_pulse(prog, prog.res_pulse, "res_pulse", ro_chs[0])

        for ro_ch in ro_chs:
            prog.declare_readout(
                ch=ro_ch,
                length=prog.us2cycles(prog.adc["ro_length"], ro_ch=ro_ch),
                freq=prog.res_pulse["freq"],
                gen_ch=res_ch,
            )

    def readout_qubit(
        self, prog: MyProgramV1, before_readout: Optional[Callable] = None
    ) -> None:
        res_ch = prog.res_pulse["ch"]
        ro_chs = prog.adc["chs"]
        if prog.ch_count[res_ch] > 1:
            set_pulse(prog, prog.res_pulse, ro_ch=ro_chs[0], waveform="res_pulse")
        if before_readout is not None:
            before_readout()
        prog.measure(
            pulse_ch=res_ch,
            adcs=ro_chs,
            adc_trig_offset=prog.us2cycles(prog.adc["trig_offset"]),
            wait=True,
            syncdelay=prog.us2cycles(prog.adc["relax_delay"]),
        )
