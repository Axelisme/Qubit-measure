from .base import BaseEFTimeProgram
from ..base import set_pulse


class EFT2EchoProgram(BaseEFTimeProgram):
    def pulse_efpi(self):
        qub_ch = self.qub_cfg["qub_ch"]
        set_pulse(self, qub_ch, self.ef_pulse["pi"], self.ef_wavform["pi"])
        self.pulse(ch=qub_ch)

    def pulse_efpi2(self):
        qub_ch = self.qub_cfg["qub_ch"]
        set_pulse(self, qub_ch, self.ef_pulse["pi2"], self.ef_wavform["pi2"])
        self.pulse(ch=qub_ch)

    def body(self):
        self.flux_ctrl.trigger()

        # pi/2 - wait - pi - wait - pi/2 sequence
        self.pulse_efpi2()
        self.sync_all()

        self.sync(self.q_rp, self.r_wait)

        self.pulse_efpi()
        self.sync_all()

        self.sync(self.q_rp, self.r_wait)

        self.pulse_efpi2()
        self.sync_all(self.us2cycles(0.05))

        self.measure_pulse()
