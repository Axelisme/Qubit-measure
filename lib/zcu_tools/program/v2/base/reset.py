from abc import ABC, abstractmethod

from myqick.asm_v2 import AveragerProgramV2

from .pulse import trigger_pulse


class AbsReset(ABC):
    @abstractmethod
    def init(self, prog: AveragerProgramV2) -> None:
        pass

    @abstractmethod
    def reset_qubit(self, prog: AveragerProgramV2) -> None:
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
    def init(self, _) -> None:
        pass

    def reset_qubit(self, _) -> None:
        pass


class PulseReset(AbsReset):
    def init(self, prog: AveragerProgramV2) -> None:
        prog.declare_pulse(prog.reset_pulse, "reset")

        if hasattr(prog, "reset_pi_pulse"):
            prog.declare_pulse(prog.reset_pi_pulse, "reset_pi")

    def reset_qubit(self, prog: AveragerProgramV2) -> None:
        trigger_pulse(prog, prog.reset_pulse, "reset")

        # if reset_pi_pulse exists, trigger it after reset
        if hasattr(prog, "reset_pi_pulse"):
            trigger_pulse(prog, prog.reset_pi_pulse, "reset_pi")


class TwoPulseReset(AbsReset):
    def init(self, prog: AveragerProgramV2) -> None:
        prog.declare_pulse(prog.reset_pulse1, "reset1")
        prog.declare_pulse(prog.reset_pulse2, "reset2")

        if hasattr(prog, "reset_pi_pulse"):
            prog.declare_pulse(prog.reset_pi_pulse, "reset_pi")

    def reset_qubit(self, prog: AveragerProgramV2) -> None:
        trigger_pulse(prog, prog.reset_pulse1, "reset1")
        trigger_pulse(prog, prog.reset_pulse2, "reset2")

        # if reset_pi_pulse exists, trigger it after reset
        if hasattr(prog, "reset_pi_pulse"):
            trigger_pulse(prog, prog.reset_pi_pulse, "reset_pi")
