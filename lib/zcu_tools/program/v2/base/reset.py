from abc import ABC, abstractmethod

from .pulse import declare_pulse


class AbsReset(ABC):
    @abstractmethod
    def init(self, prog):
        pass

    @abstractmethod
    def reset_qubit(self, prog):
        pass


def make_reset(name: str) -> AbsReset:
    if name == "none":
        return NoneReset()
    elif name == "pulse":
        return PulseReset()
    elif name == "two_pulse":
        return TwoPulseReset()
    else:
        raise ValueError(f"Unknown reset type: {name}")


class NoneReset(AbsReset):
    def init(self, _):
        pass

    def reset_qubit(self, _):
        pass


class PulseReset(AbsReset):
    DEFAULT_RESET_DELAY = 1.0  # us

    def init(self, prog):
        declare_pulse(prog, prog.reset_pulse, "reset")

    def reset_qubit(self, prog):
        reset_pulse = prog.reset_pulse
        prog.pulse(reset_pulse["ch"], "reset")
        prog.delay_auto(reset_pulse.get("delay", self.DEFAULT_RESET_DELAY))


class TwoPulseReset(AbsReset):
    DEFAULT_RESET_DELAY1 = 0.01  # us
    DEFAULT_RESET_DELAY2 = 0.1  # us

    def init(self, prog):
        declare_pulse(prog, prog.reset_pulse, "reset")
        declare_pulse(prog, prog.reset_pulse2, "reset2")

    def reset_qubit(self, prog):
        reset_pulse = prog.reset_pulse
        reset_pulse2 = prog.reset_pulse2
        prog.pulse(reset_pulse["ch"], "reset")
        prog.delay_auto(reset_pulse.get("delay1", self.DEFAULT_RESET_DELAY1))
        prog.pulse(reset_pulse2["ch"], "reset2")
        prog.delay_auto(reset_pulse2.get("delay2", self.DEFAULT_RESET_DELAY2))
