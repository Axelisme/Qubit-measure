from .base import MyProgramV2, trigger_pulse


class MuxResetRabiProgram(MyProgramV2):
    def _initialize(self, cfg) -> None:
        self.declare_pulse(self.qub_pulse, "qub_pulse")
        self.declare_pulse(self.reset_test_pulse1, "reset_test1")
        self.declare_pulse(self.reset_test_pulse2, "reset_test2")
        super()._initialize(cfg)

    def _body(self, _) -> None:
        # reset
        self.resetM.reset_qubit(self)

        # qubit pulse
        trigger_pulse(self, self.qub_pulse)

        # reset test pulse
        reset_test_pulse1 = self.reset_pulse1
        reset_test_pulse2 = self.reset_pulse2

        pre_delay = reset_test_pulse1.get("pre_delay")
        post_delay = reset_test_pulse2.get("post_delay", 0.0)

        if pre_delay is not None:
            self.delay_auto(pre_delay)

        self.pulse(reset_test_pulse1["ch"], "reset_test1")
        self.pulse(reset_test_pulse2["ch"], "reset_test2")

        if post_delay is not None:
            self.delay_auto(post_delay)

        # measure
        self.readoutM.readout_qubit(self)
