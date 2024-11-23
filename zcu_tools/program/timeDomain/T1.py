from .base import BaseTimeProgram


class T1Program(BaseTimeProgram):
    def body(self):
        self.flux_ctrl.trigger()

        # pi pulse
        self.pulse(ch=self.qub_cfg["qub_ch"])
        self.sync_all()

        # wait for specified time
        self.sync(self.q_rp, self.r_wait)

        # measure
        self.measure_pulse()
