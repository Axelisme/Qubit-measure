from .base import EFBaseTimeProgram


class EFT1Program(EFBaseTimeProgram):
    def body(self):
        self.flux_ctrl.trigger()

        # ge pi pulse
        self.pulse_ge()
        self.sync_all()

        # pi pulse
        self.pulse_ef()
        self.sync_all()

        # wait for specified time
        self.sync(self.q_rp, self.r_wait)

        # measure
        self.measure_pulse()
