from abc import ABC, abstractmethod

from myqick.asm_v2 import AveragerProgramV2

from .pulse import trigger_dual_pulse, trigger_pulse


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
    elif name == "mux_dual_pulse":
        return MuxDualPulseReset()
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
        if hasattr(prog, "reset_pi_pulse"):
            trigger_pulse(prog, prog.reset_pi_pulse, "reset_pi")


class MuxDualPulseReset(AbsReset):
    def init(self, prog: AveragerProgramV2) -> None:
        prog.declare_pulse(prog.reset_pulse1, "reset1")
        prog.declare_pulse(prog.reset_pulse2, "reset2")
        if hasattr(prog, "reset_pi_pulse"):
            prog.declare_pulse(prog.reset_pi_pulse, "reset_pi")

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

    def reset_qubit(self, prog: AveragerProgramV2) -> None:
        trigger_dual_pulse(
            prog, prog.reset_pulse1, prog.reset_pulse2, "reset1", "reset2"
        )
        if hasattr(prog, "reset_pi_pulse"):
            trigger_pulse(prog, prog.reset_pi_pulse, "reset_pi")
