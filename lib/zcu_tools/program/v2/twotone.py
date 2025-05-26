from .base import MyProgramV2


class TwoToneProgram(MyProgramV2):
    def _initialize(self, cfg) -> None:
        self.declare_pulse(self.qub_pulse, "qub_pulse")
        super()._initialize(cfg)

    def _body(self, _) -> None:
        # reset
        self.resetM.reset_qubit(self)

        # qubit pulse
        qub_pulse = self.qub_pulse

        pre_delay = qub_pulse.get("pre_delay")
        post_delay = qub_pulse.get("post_delay")

        if pre_delay is not None:
            self.delay_auto(pre_delay, ros=False)

        self.pulse(qub_pulse["ch"], "qub_pulse")

        if post_delay is not None:
            self.delay_auto(post_delay, ros=False)

        # measure
        self.readoutM.readout_qubit(self)
