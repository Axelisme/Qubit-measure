from copy import deepcopy

from ..base import MyProgramV2, declare_pulse


class T2RamseyProgram(MyProgramV2):
    def _initialize(self, cfg):
        qub_pulse = deepcopy(self.qub_pulse)
        declare_pulse(self, qub_pulse, "pi2_pulse1")
        qub_pulse["phase"] = qub_pulse["phase"] + cfg["detune"] * self.dac["t2r_length"]
        declare_pulse(self, qub_pulse, "pi2_pulse2")
        super()._initialize(cfg)

    def _body(self, _):
        # reset
        self.resetM.reset_qubit(self)

        # qub pi2 pulse
        qub_ch = self.qub_pulse["ch"]
        self.pulse(qub_ch, "pi2_pulse1", t="auto")

        # wait for specified time
        self.delay_auto(t=self.dac["t2r_length"], ros=False, tag="t2r_length")

        # qub pi2 pulse
        self.pulse(qub_ch, "pi2_pulse2", t="auto")
        self.delay_auto()

        # measure
        self.readoutM.readout_qubit(self)
