from abc import ABC, abstractmethod

from qick.asm_v1 import AcquireProgram

from .pulse import set_pulse, declare_pulse


def make_reset(name: str):
    if name == "none":
        return NoneReset()
    elif name == "pulse":
        return PulseReset()
    else:
        raise ValueError(f"Unknown reset type: {name}")


class AbsReset(ABC):
    @abstractmethod
    def init(self, prog: AcquireProgram):
        pass

    @abstractmethod
    def reset_qubit(self, prog: AcquireProgram):
        pass


class NoneReset(AbsReset):
    def init(self, prog: AcquireProgram):
        pass

    def reset_qubit(self, prog: AcquireProgram):
        pass


class PulseReset(AbsReset):
    RESET_DELAY = 1.0

    def init(self, prog: AcquireProgram):
        declare_pulse(prog, prog.reset_pulse, waveform="reset")

    def reset_qubit(self, prog: AcquireProgram):
        reset_pulse = prog.reset_pulse
        if prog.ch_count[reset_pulse["ch"]] > 1:
            set_pulse(prog, reset_pulse, waveform="reset")
        prog.pulse(ch=reset_pulse["ch"])
        prog.sync_all(prog.us2cycles(self.RESET_DELAY))
