from abc import ABC, abstractmethod

from qick.asm_v2 import AveragerProgramV2

from .pulse import declare_pulse


def make_reset(name: str):
    if name == "none":
        return NoneReset()
    elif name == "pulse":
        return PulseReset()
    else:
        raise ValueError(f"Unknown reset type: {name}")


class AbsReset(ABC):
    @abstractmethod
    def init(self, prog: AveragerProgramV2):
        pass

    @abstractmethod
    def reset_qubit(self, prog: AveragerProgramV2):
        pass


class NoneReset(AbsReset):
    def init(self, _: AveragerProgramV2):
        pass

    def reset_qubit(self, _: AveragerProgramV2):
        pass


class PulseReset(AbsReset):
    DEFAULT_RESET_DELAY = 1.0  # us

    def init(self, prog: AveragerProgramV2):
        declare_pulse(prog, prog.reset_pulse, "reset")

    def reset_qubit(self, prog: AveragerProgramV2):
        reset_pulse = prog.reset_pulse
        prog.pulse(reset_pulse["ch"], "reset")
        prog.sync_all(reset_pulse.get("delay", self.DEFAULT_RESET_DELAY))
