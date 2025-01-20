from ..base import MyRAveragerProgram, SYNC_TIME
from ..twotone import declare_pulse, set_pulse  # noqa


class TimeProgram(MyRAveragerProgram):
    def declare_wait_reg(self, ch):
        self.q_rp = self.ch_page(ch)
        self.r_wait = 3
        self.regwi(self.q_rp, self.r_wait, self.cfg["start"])

    def parse_sweep(self):
        sweep_cfg = self.sweep_cfg
        self.cfg["start"] = self.us2cycles(sweep_cfg["start"])
        self.cfg["step"] = self.us2cycles(sweep_cfg["step"])
        self.cfg["expts"] = sweep_cfg["expts"]

    def initialize(self):
        self.parse_sweep()
        self.resetM.init(self)
        self.readoutM.init(self)
        declare_pulse(self, self.qub_pulse, "qub_pulse")
        self.declare_wait_reg(self.qub_pulse["ch"])

        self.synci(SYNC_TIME)

    def update(self):
        self.mathi(self.q_rp, self.r_wait, self.r_wait, "+", self.cfg["step"])
