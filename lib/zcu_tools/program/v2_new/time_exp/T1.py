from ..base import MyProgramV2


class T1Program(MyProgramV2):
    def _initialize(self, cfg):
        self.declare_pulse(self.qub_pulse, "pi_pulse")
        super()._initialize(cfg)

    def _body(self, _):
        # reset
        self.resetM.reset_qubit(self)

        # qub pi pulse
        self.pulse(self.qub_pulse["ch"], "pi_pulse")

        # wait for specified time
        self.delay_auto(t=self.dac["t1_length"], ros=False, tag="t1_length")

        # measure
        self.readoutM.readout_qubit(self)
