from abc import ABC, abstractmethod

from myqick.asm_v2 import AveragerProgramV2

from .pulse import trigger_pulse


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
    def init(self, prog: AveragerProgramV2):
        prog.declare_pulse(prog.reset_pulse, "reset")

    def reset_qubit(self, prog: AveragerProgramV2):
        trigger_pulse(prog, prog.reset_pulse)


class TwoPulseReset(AbsReset):
    def init(self, prog: AveragerProgramV2):
        prog.declare_pulse(prog.reset_pulse1, "reset1")
        prog.declare_pulse(prog.reset_pulse2, "reset2")

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

    def reset_qubit(self, prog: AveragerProgramV2):
        reset_pulse1 = prog.reset_pulse1
        reset_pulse2 = prog.reset_pulse2

        pre_delay = reset_pulse1.get("pre_delay")
        post_delay = reset_pulse2.get("post_delay")

        if pre_delay is not None:
            prog.delay_auto(pre_delay)

        prog.pulse(reset_pulse1["ch"], "reset1")
        prog.pulse(reset_pulse2["ch"], "reset2")

        if post_delay is not None:
            prog.delay_auto(post_delay)
