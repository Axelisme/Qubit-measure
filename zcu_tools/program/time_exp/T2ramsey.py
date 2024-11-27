from .base import BaseTimeProgram


class T2RamseyProgram(BaseTimeProgram):
    def body(self):
        # qub_cfg = self.qub_cfg
        qub_ch = self.qub_cfg["qub_ch"]

        self.flux_ctrl.trigger()

        # pi/2 - wait - pi/2 sequence
        self.pulse(ch=qub_ch)
        self.sync_all()

        self.sync(self.q_rp, self.r_wait)

        self.pulse(ch=qub_ch)
        self.sync_all(self.us2cycles(0.05))

        self.measure_pulse()
