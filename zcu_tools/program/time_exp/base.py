from qick import RAveragerProgram

from ..base import BaseTwoToneProgram


class TimeProgram(RAveragerProgram, BaseTwoToneProgram):
    def parse_cfg(self):
        BaseTwoToneProgram.parse_cfg(self)

        sweep_cfg = self.cfg["sweep"]
        self.cfg["start"] = self.us2cycles(sweep_cfg["start"])
        self.cfg["step"] = self.us2cycles(sweep_cfg["step"])
        self.cfg["expts"] = sweep_cfg["expts"]

    def setup_waittime(self):
        self.q_rp = self.ch_page(self.qub_pulse["ch"])
        self.r_wait = 3
        self.regwi(self.q_rp, self.r_wait, self.cfg["start"])

    def initialize(self):
        self.parse_cfg()
        self.setup_readout()
        self.setup_qubit()
        self.setup_waittime()

        self.synci(200)

    def update(self):
        self.mathi(self.q_rp, self.r_wait, self.r_wait, "+", self.cfg["step"])
