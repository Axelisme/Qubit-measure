from .base import MyProgramV2

PULSE_DELAY = 0.05  # us


class TwoToneProgram(MyProgramV2):
    def _body(self, _):
        # reset
        self.resetM.reset_qubit(self)

        # qubit pulse
        self.pulse(self.qub_pulse["ch"], "qub_pulse")
        self.delay_auto(PULSE_DELAY)  # type: ignore

        # measure
        self.readoutM.readout_qubit(self)
