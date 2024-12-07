from qick import AveragerProgram, RAveragerProgram

from .base import BaseOneToneProgram


class OneToneProgram(AveragerProgram, BaseOneToneProgram):
    def initialize(self):
        BaseOneToneProgram.initialize(self)

    def body(self):
        BaseOneToneProgram.body(self)


class RGainOnetoneProgram(RAveragerProgram, BaseOneToneProgram):
    def parse_cfg(self):
        BaseOneToneProgram.parse_cfg(self)

        sweep_cfg = self.cfg["sweep"]
        self.cfg["start"] = sweep_cfg["start"]
        self.cfg["step"] = sweep_cfg["step"]
        self.cfg["expts"] = sweep_cfg["expts"]

        self.res_pulse["gain"] = self.cfg["start"]

    def setup_gain_reg(self):
        # setup gain register
        ch = self.res_pulse["ch"]
        self.r_rp = self.ch_page(ch)
        self.r_gain = self.sreg(ch, "gain")
        self.regwi(self.r_rp, self.r_gain, self.cfg["start"])

    def initialize(self):
        self.parse_cfg()
        self.setup_readout()
        self.setup_gain_reg()

        self.synci(200)

    def body(self):
        BaseOneToneProgram.body(self)

    def update(self):
        self.mathi(self.r_rp, self.r_gain, self.r_gain, "+", self.cfg["step"])
