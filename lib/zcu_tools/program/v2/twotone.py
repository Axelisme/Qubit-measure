from .base import MyProgramV2


class TwoToneProgram(MyProgramV2):
    PULSE_DELAY = 0.01  # us

    def _initialize(self, cfg) -> None:
        self.declare_pulse(self.qub_pulse, "qub_pulse")
        super()._initialize(cfg)

    def _body(self, _) -> None:
        # reset
        self.resetM.reset_qubit(self)

        # qubit pulse
        qub_pulse = self.qub_pulse

        self.delay_auto(qub_pulse.get("pre_delay", self.PULSE_DELAY), ros=False)
        self.pulse(qub_pulse["ch"], "qub_pulse")
        self.delay_auto(qub_pulse.get("post_delay", self.PULSE_DELAY), ros=False)

        # measure
        self.readoutM.readout_qubit(self)
