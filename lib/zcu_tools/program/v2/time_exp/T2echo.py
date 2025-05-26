from copy import deepcopy

from ..base import MyProgramV2


class T2EchoProgram(MyProgramV2):
    PULSE_DELAY = 0.01  # us

    def _initialize(self, cfg):
        self.declare_pulse(self.pi_pulse, "pi_pulse")

        pi2_pulse = deepcopy(self.pi2_pulse)
        self.declare_pulse(pi2_pulse, "pi2_pulse1")
        pi2_pulse["phase"] = pi2_pulse["phase"] + 360 * cfg["detune"] * (
            2 * self.dac["t2e_half"] + self.pi_pulse["length"]
        )
        self.declare_pulse(pi2_pulse, "pi2_pulse2")
        super()._initialize(cfg)

    def _body(self, _):
        # reset
        self.resetM.reset_qubit(self)

        # qub pi2 pulse
        self.pulse(self.pi2_pulse["ch"], "pi2_pulse1", t="auto")

        # wait for specified time
        self.delay_auto(t=self.dac["t2e_half"], ros=False, tag="t2e_half")

        # qub pi pulse
        self.pulse(self.pi_pulse["ch"], "pi_pulse", t="auto")

        # wait for specified time
        self.delay_auto(t=self.dac["t2e_half"], ros=False)

        # qub pi2 pulse
        self.pulse(self.pi2_pulse["ch"], "pi2_pulse2", t="auto")
        self.delay_auto(self.pi2_pulse.get("post_delay", self.PULSE_DELAY), ros=False)

        # measure
        self.readoutM.readout_qubit(self)
