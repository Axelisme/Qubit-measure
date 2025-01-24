from ..base import MyRAveragerProgram, SYNC_TIME
from ..twotone import declare_pulse, set_pulse, PULSE_DELAY  # noqa


class TimeProgram(MyRAveragerProgram):
    def declare_wait_reg(self, ch):
        self.q_rp = self.ch_page(ch)
        self.q_wait = 3
        self.regwi(self.q_rp, self.q_wait, self.us2cycles(self.cfg["start"]))
        self.q_step = self.us2cycles(self.cfg["step"])

    def initialize(self):
        self.resetM.init(self)
        self.readoutM.init(self)
        declare_pulse(self, self.qub_pulse, "qub_pulse")
        self.declare_wait_reg(self.qub_pulse["ch"])

        self.synci(SYNC_TIME)

    def update(self):
        self.mathi(self.q_rp, self.q_wait, self.q_wait, "+", self.q_step)
