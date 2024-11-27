from qick import AveragerProgram
from .base import BaseEFProgram


class EFTwoToneProgram(AveragerProgram, BaseEFProgram):
    def initialize(self):
        return BaseEFProgram.initialize(self)

    def body(self):
        self.flux_ctrl.trigger()

        # ge pi pulse
        self.pulse_ge()
        self.sync_all()

        # qubit pulse
        self.pulse_ef()
        self.sync_all(self.us2cycles(0.05))

        # measure
        self.measure_pulse()
