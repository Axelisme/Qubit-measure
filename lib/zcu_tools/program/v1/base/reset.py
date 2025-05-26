from abc import ABC, abstractmethod

from myqick.asm_v1 import QickProgram

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
    def init(self, prog: QickProgram):
        pass

    @abstractmethod
    def reset_qubit(self, prog: QickProgram):
        pass


class NoneReset(AbsReset):
    def init(self, prog: QickProgram):
        pass

    def reset_qubit(self, prog: QickProgram):
        pass


class PulseReset(AbsReset):
    def init(self, prog: QickProgram):
        declare_pulse(prog, prog.reset_pulse, waveform="reset")

    def reset_qubit(self, prog: QickProgram):
        reset_pulse = prog.reset_pulse

        pre_delay = reset_pulse.get("pre_delay")
        post_delay = reset_pulse.get("post_delay")

        if pre_delay is not None:
            prog.sync_all(prog.us2cycles(pre_delay))

        if prog.ch_count[reset_pulse["ch"]] > 1:
            set_pulse(prog, reset_pulse, waveform="reset")
        prog.pulse(ch=reset_pulse["ch"])

        if post_delay is not None:
            prog.sync_all(prog.us2cycles(post_delay))


class TwoPulseReset(AbsReset):
    def init(self, prog: QickProgram):
        declare_pulse(prog, prog.reset_pulse1, waveform="reset1")
        declare_pulse(prog, prog.reset_pulse2, waveform="reset2")

        if prog.reset_pulse1["ch"] == prog.reset_pulse2["ch"]:
            raise ValueError(
                "reset_pulse1 and reset_pulse2 cannot have the same channel"
            )

        if prog.reset_pulse1["length"] != prog.reset_pulse2["length"]:
            raise ValueError("reset_pulse1 and reset_pulse2 must have the same length")

        if prog.reset_pulse1.get("post_delay") is not None:
            raise ValueError("reset_pulse1 cannot have a post_delay")

        if prog.reset_pulse2.get("pre_delay") is not None:
            raise ValueError("reset_pulse2 cannot have a pre_delay")

    def reset_qubit(self, prog: QickProgram):
        reset_pulse1 = prog.reset_pulse1
        reset_pulse2 = prog.reset_pulse2

        pre_delay = reset_pulse1.get("pre_delay")
        post_delay = reset_pulse2.get("post_delay")

        if pre_delay is not None:
            prog.sync_all(prog.us2cycles(pre_delay))

        if prog.ch_count[reset_pulse1["ch"]] > 1:
            set_pulse(prog, reset_pulse1, waveform="reset1")
        prog.pulse(ch=reset_pulse1["ch"])

        if prog.ch_count[reset_pulse2["ch"]] > 1:
            set_pulse(prog, reset_pulse2, waveform="reset2")
        prog.pulse(ch=reset_pulse2["ch"])

        if post_delay is not None:
            prog.sync_all(prog.us2cycles(post_delay))
