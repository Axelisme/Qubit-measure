from .base import BaseTimeProgram
from ..base import set_pulse


class T2EchoProgram(BaseTimeProgram):
    def body(self):
        qub_ch = self.qub_cfg["qub_ch"]
        pi_cfg = self.qub_pulse["pi"]
        pi2_cfg = self.qub_pulse["pi2"]
        pi_wavform = self.qub_wavform["pi"]
        pi2_wavform = self.qub_wavform["pi2"]

        self.flux_ctrl.trigger()

        # pi/2 - wait - pi - wait - pi/2 sequence
        set_pulse(self, qub_ch, pi2_cfg, pi2_wavform)
        self.pulse(ch=qub_ch)
        self.sync_all()

        self.sync(self.q_rp, self.r_wait)

        set_pulse(self, qub_ch, pi_cfg, pi_wavform)
        self.pulse(ch=qub_ch)
        self.sync_all()

        self.sync(self.q_rp, self.r_wait)

        set_pulse(self, qub_ch, pi2_cfg, pi2_wavform)
        self.pulse(ch=qub_ch)
        self.sync_all(self.us2cycles(0.05))

        self.measure_pulse()
