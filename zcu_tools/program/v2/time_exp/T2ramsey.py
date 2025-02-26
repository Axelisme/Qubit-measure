from ..base import MyProgramV2, declare_pulse


class T2RamseyProgram(MyProgramV2):
    def _initialize(self, cfg):
        declare_pulse(self, self.qub_pulse, "pi2_pulse")
        super()._initialize(cfg)

    def _body(self, _):
        # reset
        self.resetM.reset_qubit(self)

        # qub pi2 pulse
        self.pulse(self.qub_pulse["ch"], "pi2_pulse", t=None)  

        # wait for specified time
        self.delay_auto(t=self.dac["t2r_length"], ros=False, tag="T2r")

        # qub pi2 pulse
        self.pulse(self.qub_pulse["ch"], "pi2_pulse", t=None)  
        self.delay_auto()

        # measure
        self.readoutM.readout_qubit(self)
