from abc import ABC, abstractmethod


from myqick.asm_v2 import AveragerProgramV2


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
        return MuxPulseReset()
    else:
        raise ValueError(f"Unknown reset type: {name}")


class NoneReset(AbsReset):
    def init(self, _):
        pass

    def reset_qubit(self, _):
        pass


class PulseReset(AbsReset):
    DEFAULT_RESET_DELAY = 0.1  # us

    def init(self, prog: AveragerProgramV2):
        prog.declare_pulse(prog.reset_pulse, "reset")

    def reset_qubit(self, prog: AveragerProgramV2):
        reset_pulse = prog.reset_pulse
        prog.pulse(reset_pulse["ch"], "reset")
        prog.delay_auto(reset_pulse.get("post_delay", self.DEFAULT_RESET_DELAY))


class MuxPulseReset(AbsReset):
    DEFAULT_RESET_DELAY = 0.1  # us

    def init(self, prog: AveragerProgramV2):
        prog.declare_pulse(prog.reset_pulse1, "mux_reset")
        prog.declare_pulse(prog.reset_pulse2, "mux_reset2")

        if prog.reset_pulse1["ch"] == prog.reset_pulse2["ch"]:
            # TODO: Add support for mux reset on the same channel
            raise ValueError(
                "MuxPulseReset requires different channels for both pulses."
            )

        length1 = prog.reset_pulse1["length"]
        length2 = prog.reset_pulse2["length"]
        if length1 != length2:
            raise ValueError("MuxPulseReset requires equal length for both pulses.")

        if prog.reset_pulse1["post_delay"] != 0.0:
            raise ValueError(
                "MuxPulseReset requires post_delay of result_pulse1 to be 0.0."
            )

    def reset_qubit(self, prog: AveragerProgramV2):
        reset_pulse1 = prog.reset_pulse1
        reset_pulse2 = prog.reset_pulse2
        prog.pulse(reset_pulse1["ch"], "mux_reset")
        prog.pulse(reset_pulse2["ch"], "mux_reset2")
        prog.delay_auto(reset_pulse2.get("post_delay", self.DEFAULT_RESET_DELAY))
