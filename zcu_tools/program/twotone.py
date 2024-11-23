from qick import AveragerProgram

from .base import BaseTwoToneProgram


class TwoToneProgram(AveragerProgram, BaseTwoToneProgram):
    def initialize(self):
        return BaseTwoToneProgram.initialize(self)

    def body(self):
        self.flux_ctrl.trigger()

        # qubit pulse
        self.pulse(ch=self.qub_cfg["qub_ch"])
        self.sync_all(self.us2cycles(0.05))

        # measure
        self.measure_pulse()
