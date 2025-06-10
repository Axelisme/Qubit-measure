from .base import MyProgramV2, trigger_pulse


class TwoToneProgram(MyProgramV2):
    def _initialize(self, cfg) -> None:
        self.declare_pulse(self.qub_pulse, "qub_pulse")
        super()._initialize(cfg)

    def _body(self, _) -> None:
        # reset
        self.resetM.reset_qubit(self)

        # qubit pulse
        trigger_pulse(self, self.qub_pulse, "qub_pulse")

        # measure
        self.readoutM.readout_qubit(self)
