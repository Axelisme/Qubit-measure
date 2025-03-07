from ..base import SYNC_TIME, MyRAveragerProgram
from ..twotone import declare_pulse, set_pulse


class T1Program(MyRAveragerProgram):
    def declare_wait_reg(self, q_rp):
        self.q_wait = 3
        self.regwi(q_rp, self.q_wait, self.us2cycles(self.cfg["start"]))
        self.q_wait_s = self.us2cycles(self.cfg["step"])

    def initialize(self):
        self.resetM.init(self)
        self.readoutM.init(self)
        declare_pulse(self, self.qub_pulse, "qub_pulse")

        qub_ch = self.qub_pulse["ch"]
        self.q_rp = self.ch_page(qub_ch)

        self.declare_wait_reg(self.q_rp)

        self.synci(SYNC_TIME)

    def update(self):
        self.mathi(self.q_rp, self.q_wait, self.q_wait, "+", self.q_wait_s)

    def body(self):
        # reset
        self.resetM.reset_qubit(self)

        # pi pulse
        ch = self.qub_pulse["ch"]
        if self.ch_count[ch] > 1:
            set_pulse(self, self.qub_pulse, waveform="qub_pulse")
        self.pulse(ch=ch)
        self.sync_all()

        # wait for specified time
        self.sync(self.q_rp, self.q_wait)

        # measure
        self.readoutM.readout_qubit(self)
