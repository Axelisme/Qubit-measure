from .base import EFProgram


class EFTwoToneProgram(EFProgram):
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
