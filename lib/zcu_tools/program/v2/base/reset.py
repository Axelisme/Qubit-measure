from abc import ABC, abstractmethod

from qick.asm_v2 import AveragerProgramV2

from .pulse import declare_pulse


class AbsReset(ABC):
    @abstractmethod
    def init(self, prog: AveragerProgramV2):
        pass

    @abstractmethod
    def reset_qubit(self, prog: AveragerProgramV2):
        pass


def make_reset(name: str) -> AbsReset:
    if name == "none":
        return NoneReset()
    elif name == "pulse":
        return PulseReset()
    else:
        raise ValueError(f"Unknown reset type: {name}")


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
        prog.delay_auto(reset_pulse.get("delay", self.DEFAULT_RESET_DELAY))
