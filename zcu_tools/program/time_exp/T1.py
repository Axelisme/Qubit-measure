from .base import TimeProgram


class T1Program(TimeProgram):
    def body(self):
        # pi pulse
        self.pulse(ch=self.qub_pulse["ch"])
        self.sync_all()

        # wait for specified time
        self.sync(self.q_rp, self.r_wait)

        # measure
        self.measure_pulse()
