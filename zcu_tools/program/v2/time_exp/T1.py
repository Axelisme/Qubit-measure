from ..base import MyProgramV2, declare_pulse


class T1Program(MyProgramV2):
    def _initialize(self, cfg):
        declare_pulse(self, self.qub_pulse, "pi_pulse")
        super()._initialize(cfg)

    def _body(self, _):
        # reset
        self.resetM.reset_qubit(self)

        # qub pi pulse
        self.pulse(self.qub_pulse["ch"], "pi_pulse")

        # wait for specified time
        self.delay_auto(t=self.dac["t1_length"], ros=False, tag="T1")

        # measure
        self.readoutM.readout_qubit(self)
