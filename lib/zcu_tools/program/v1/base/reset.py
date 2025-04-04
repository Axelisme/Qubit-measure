from abc import ABC, abstractmethod

from .pulse import declare_pulse, set_pulse


def make_reset(name: str):
    if name == "none":
        return NoneReset()
    elif name == "pulse":
        return PulseReset()
    elif name == "two_pulse":
        return TwoPulseReset()
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
    DEFAULT_RESET_DELAY = 0.1  # us

    def init(self, prog):
        declare_pulse(prog, prog.reset_pulse, waveform="reset")

    def reset_qubit(self, prog):
        reset_pulse = prog.reset_pulse
        if prog.ch_count[reset_pulse["ch"]] > 1:
            set_pulse(prog, reset_pulse, waveform="reset")
        prog.pulse(ch=reset_pulse["ch"])
        delay = reset_pulse.get("delay", self.DEFAULT_RESET_DELAY)
        prog.sync_all(prog.us2cycles(delay))


class TwoPulseReset(AbsReset):
    DEFAULT_RESET_DELAY1 = 0.01  # us
    DEFAULT_RESET_DELAY2 = 0.1  # us

    def init(self, prog):
        declare_pulse(prog, prog.reset_pulse, waveform="reset")
        declare_pulse(prog, prog.reset_pulse2, waveform="reset2")

    def reset_qubit(self, prog):
        reset_pulse = prog.reset_pulse
        reset_pulse2 = prog.reset_pulse2

        if prog.ch_count[reset_pulse["ch"]] > 1:
            set_pulse(prog, reset_pulse, waveform="reset")
        prog.pulse(ch=reset_pulse["ch"])
        delay1 = reset_pulse.get("delay1", self.DEFAULT_RESET_DELAY1)
        prog.sync_all(prog.us2cycles(delay1))

        if prog.ch_count[reset_pulse2["ch"]] > 1:
            set_pulse(prog, reset_pulse2, waveform="reset2")
        prog.pulse(ch=reset_pulse2["ch"])
        delay2 = reset_pulse2.get("delay2", self.DEFAULT_RESET_DELAY2)
        prog.sync_all(prog.us2cycles(delay2))
