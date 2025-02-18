from .base import MyProgramV2, declare_pulse

PULSE_DELAY = 0.01  # us


class TwoToneProgram(MyProgramV2):
    def _initialize(self, cfg):
        declare_pulse(self, self.qub_pulse, "qub_pulse")
        super()._initialize(cfg)

    def _body(self, _):
        # reset
        self.resetM.reset_qubit(self)

        # qubit pulse
        self.pulse(self.qub_pulse["ch"], "qub_pulse")
        self.delay_auto(PULSE_DELAY)  # type: ignore

        # measure
        self.readoutM.readout_qubit(self)
