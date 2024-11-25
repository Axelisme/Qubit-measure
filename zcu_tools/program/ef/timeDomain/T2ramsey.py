from .base import BaseEFTimeProgram


class EFT2RamseyProgram(BaseEFTimeProgram):
    def body(self):
        self.flux_ctrl.trigger()

        # ge pulse
        self.pulse_ge()

        # pi/2 - wait - pi/2 sequence
        self.pulse_ef()
        self.sync_all()

        self.sync(self.q_rp, self.r_wait)

        self.pulse_ef()
        self.sync_all(self.us2cycles(0.05))

        self.measure_pulse()
