from .base import MyProgramV2, declare_pulse


class TwoToneProgram(MyProgramV2):
    PULSE_DELAY = 0.01  # us

    def _initialize(self, cfg):
        declare_pulse(self, self.qub_pulse, "qub_pulse")
        super()._initialize(cfg)

    def _body(self, _):
        # reset
        self.resetM.reset_qubit(self)

        # qubit pulse
        qub_pulse = self.qub_pulse
        self.delay_auto(qub_pulse.get("pre_delay", self.PULSE_DELAY))
        self.pulse(qub_pulse["ch"], "qub_pulse")
        self.delay_auto(qub_pulse.get("post_delay", self.PULSE_DELAY))

        # measure
        self.readoutM.readout_qubit(self)
