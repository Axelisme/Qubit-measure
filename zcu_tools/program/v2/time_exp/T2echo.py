from ..base import MyProgramV2, declare_pulse


class T2EchoProgram(MyProgramV2):
    def _initialize(self, cfg):
        declare_pulse(self, self.pi_pulse, "pi_pulse")
        declare_pulse(self, self.pi2_pulse, "pi2_pulse")
        super()._initialize(cfg)

    def _body(self, _):
        # reset
        self.resetM.reset_qubit(self)

        # qub pi2 pulse
        self.pulse(self.pi2_pulse["ch"], "pi2_pulse", t=None)  # type: ignore

        # wait for specified time
        self.delay_auto(t=self.sweep_cfg["length"], ros=False, tag="T2e")

        # qub pi pulse
        self.pulse(self.pi_pulse["ch"], "pi_pulse", t=None)  # type: ignore

        # wait for specified time
        self.delay_auto(t=self.sweep_cfg["length"], ros=False, tag="T2e")

        # qub pi2 pulse
        self.pulse(self.pi2_pulse["ch"], "pi2_pulse", t=None)  # type: ignore
        self.delay_auto()

        # measure
        self.readoutM.readout_qubit(self)
