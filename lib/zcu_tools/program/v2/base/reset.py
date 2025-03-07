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
