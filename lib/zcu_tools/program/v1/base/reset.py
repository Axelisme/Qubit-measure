from abc import ABC, abstractmethod

from .pulse import declare_pulse, set_pulse


def make_reset(name: str):
    if name == "none":
        return NoneReset()
    elif name == "pulse":
        return PulseReset()
    else:
        raise ValueError(f"Unknown reset type: {name}")


class AbsReset(ABC):
    @abstractmethod
    def init(self, prog):
        pass

    @abstractmethod
    def reset_qubit(self, prog):
        pass


class NoneReset(AbsReset):
    def init(self, prog):
        pass

    def reset_qubit(self, prog):
        pass


class PulseReset(AbsReset):
    DEFAULT_RESET_DELAY = 1.0

    def init(self, prog):
        declare_pulse(prog, prog.reset_pulse, waveform="reset")

    def reset_qubit(self, prog):
        reset_pulse = prog.reset_pulse
        if prog.ch_count[reset_pulse["ch"]] > 1:
            set_pulse(prog, reset_pulse, waveform="reset")
        prog.pulse(ch=reset_pulse["ch"])
        delay = reset_pulse.get("delay", self.DEFAULT_RESET_DELAY)
        prog.sync_all(prog.us2cycles(delay))
