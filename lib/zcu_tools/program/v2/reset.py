from .base import MyProgramV2, trigger_pulse


class ResetProgram(MyProgramV2):
    def _initialize(self, cfg) -> None:
        self.declare_pulse(self.qub_pulse, "qub_pulse")
        self.declare_pulse(self.reset_test_pulse, "reset_test")
        super()._initialize(cfg)

    def _body(self, _) -> None:
        # reset
        self.resetM.reset_qubit(self)

        # qubit pulse
        trigger_pulse(self, self.qub_pulse, "qub_pulse")

        # reset test pulse
        trigger_pulse(self, self.reset_test_pulse, "reset_test")

        # measure
        self.readoutM.readout_qubit(self)


class TwoPulseResetProgram(MyProgramV2):
    def _initialize(self, cfg) -> None:
        self.declare_pulse(self.qub_pulse, "qub_pulse")
        self.declare_pulse(self.reset_test_pulse1, "reset_test1")
        self.declare_pulse(self.reset_test_pulse2, "reset_test2")

        super()._initialize(cfg)

    def _body(self, _) -> None:
        # reset
        self.resetM.reset_qubit(self)

        # qubit pulse
        trigger_pulse(self, self.qub_pulse, "qub_pulse")

        # reset test pulse
        trigger_pulse(self, self.reset_test_pulse1, "reset_test1")
        trigger_pulse(self, self.reset_test_pulse2, "reset_test2")

        # measure
        self.readoutM.readout_qubit(self)
