from qick import AveragerProgram

from .base import BaseOneToneProgram


class OnetoneProgram(AveragerProgram, BaseOneToneProgram):
    def initialize(self):
        return BaseOneToneProgram.initialize(self)

    def body(self):
        self.flux_ctrl.trigger()

        self.measure_pulse()
