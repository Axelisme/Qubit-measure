from .base import TimeProgram


class T2RamseyProgram(TimeProgram):
    def body(self):
        qub_ch = self.qub_pulse["ch"]

        # pi/2 - wait - pi/2 sequence
        self.pulse(ch=qub_ch)
        self.sync_all()

        self.sync(self.q_rp, self.r_wait)

        self.pulse(ch=qub_ch)
        self.sync_all(self.us2cycles(0.05))

        self.measure_pulse()
